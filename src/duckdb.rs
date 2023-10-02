use std::borrow::Cow;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use arrow::array as aa;
use duckdb::Connection;
use duckdb::types::ToSqlOutput;

use crate::arrow::{Array, FromArrow, Iter};

pub use duckdb::types::ValueRef;
pub use duckdb::types::Null;

pub trait Spec: Sized {
    type Params<'a>: Params = ();
    type Results: Results = ();
    const SQL: &'static str;

    fn execute(conn: &Connection, params: Self::Params<'_>) -> Result<usize> {
        Statement::<Self>::new(conn)?.execute(params)
    }

    fn query(conn: &Connection, params: Self::Params<'_>) -> Result<Vec<Self::Results>> {
        Query::<Self>::new(conn)?.query(params)?.try_collect()
    }

    fn query_row(conn: &Connection, params: Self::Params<'_>) -> Result<Self::Results> {
        QueryRow::<Self>::new(conn)?.query_row(params)
    }
}

pub struct HasTable;

impl Spec for HasTable {
    type Params<'a> = &'a str;
    type Results = bool;

    const SQL: &'static str = "
        select count(*) == 1
        from information_schema.tables
        where table_name = $1
    ";
}

// -----------------------------------------------------------------------------

pub struct Statement<'conn, S> {
    duck: duckdb::Statement<'conn>,
    phantom: PhantomData<S>,
}

impl<'conn, S: Spec> Statement<'conn, S> {
    pub fn new(conn: &'conn Connection) -> Result<Statement<'conn, S>> {
        let duck = conn.prepare(S::SQL).context(S::SQL)?;
        Ok(Statement { duck, phantom: PhantomData })
    }

    pub fn execute(&mut self, params: S::Params<'_>) -> Result<usize> {
        params.bind(&mut self.duck)
            .and_then(|()| self.duck.raw_execute())
            .context(S::SQL)
    }
}

// -----------------------------------------------------------------------------

pub struct Query<'conn, S>(Statement<'conn, S>);

impl<'conn, S: Spec> Query<'conn, S> {
    pub fn new(conn: &'conn Connection) -> Result<Query<'conn, S>> {
        Ok(Query(Statement::new(conn)?))
    }

    pub fn query(&mut self, params: S::Params<'_>) -> Result<QueryResults<'_, 'conn, S::Results>> {
        self.0.execute(params)?;

        Ok(QueryResults {
            statement: &mut self.0.duck,
            rows: None,
            sql: S::SQL,
        })
    }
}

// -----------------------------------------------------------------------------

pub struct QueryRow<'conn, S>(Query<'conn, S>);

impl<'conn, S: Spec> QueryRow<'conn, S> {
    pub fn new(conn: &'conn Connection) -> Result<QueryRow<'conn, S>> {
        Ok(QueryRow(Query::new(conn)?))
    }

    pub fn query_row(&mut self, params: S::Params<'_>) -> Result<S::Results> {
        match self.0.query(params)?.next() {
            Some(row) => row,
            None => Err(duckdb::Error::QueryReturnedNoRows).context(S::SQL),
        }
    }
}

// -----------------------------------------------------------------------------

pub struct QueryResults<'a, 'conn, R: Results> {
    statement: &'a mut duckdb::Statement<'conn>,
    rows: Option<R::Iter>,
    sql: &'static str,
}

impl<'a, 'conn, R: Results> Iterator for QueryResults<'a, 'conn, R> {
    type Item = Result<R>;

    fn next(&mut self) -> Option<Result<R>> {
        loop {
            if let Some(iter) = &mut self.rows {
                if let Some(x) = iter.next() {
                    return Some(Ok(x));
                } else {
                    self.rows = None;
                }
            }

            match R::batch_iter(self.statement.step()?) {
                Ok(rows) => self.rows = Some(rows),
                Err(e) => return Some(Err(e).context(self.sql)),
            }
        }
    }
}

// -----------------------------------------------------------------------------

pub trait Results {
    type Iter: Iterator<Item=Self>;

    fn batch_iter(array: aa::StructArray) -> Result<Self::Iter>;
}

impl<T: FromArrow> Results for T {
    type Iter = Iter<T>;

    fn batch_iter(array: aa::StructArray) -> Result<Self::Iter> {
        if aa::Array::null_count(&array) == 0 {
            Ok(Iter::new(T::Array::from_arrow_batch(&array.into())?))
        } else {
            Err(anyhow!("Result batch contains toplevel nulls"))
        }
    }
}

// -----------------------------------------------------------------------------

pub trait ToSql {
    fn to_sql(&self) -> ValueRef;
}

impl ToSql for Null {
    fn to_sql(&self) -> ValueRef {
        ValueRef::Null
    }
}

impl ToSql for str {
    fn to_sql(&self) -> ValueRef {
        ValueRef::Text(self.as_bytes())
    }
}

impl ToSql for [u8] {
    fn to_sql(&self) -> ValueRef {
        ValueRef::Blob(self)
    }
}

impl ToSql for String {
    fn to_sql(&self) -> ValueRef {
        ValueRef::Text(self.as_bytes())
    }
}

impl ToSql for Vec<u8> {
    fn to_sql(&self) -> ValueRef {
        ValueRef::Blob(&self)
    }
}

impl<T: ToSql> ToSql for Option<T> {
    fn to_sql(&self) -> ValueRef {
        match self {
            Some(x) => x.to_sql(),
            None => ValueRef::Null,
        }
    }
}

impl<T: ToSql + ?Sized> ToSql for &T {
    fn to_sql(&self) -> ValueRef {
        T::to_sql(self)
    }
}

impl<T: ToSql + ?Sized> ToSql for Box<T> {
    fn to_sql(&self) -> ValueRef {
        T::to_sql(self)
    }
}

impl<T: ToSql + ?Sized> ToSql for Rc<T> {
    fn to_sql(&self) -> ValueRef {
        T::to_sql(self)
    }
}

impl<T: ToSql + ?Sized> ToSql for Arc<T> {
    fn to_sql(&self) -> ValueRef {
        T::to_sql(self)
    }
}

impl<T: ToSql + ToOwned + ?Sized> ToSql for Cow<'_, T> {
    fn to_sql(&self) -> ValueRef {
        T::to_sql(self)
    }
}

macro_rules! impl_to_sql {
    { $( $c:ident($p:ty), )+ } => { $(
        impl ToSql for $p {
            fn to_sql(&self) -> ValueRef {
                ValueRef::$c(*self)
            }
        }
    )+ };
}

impl_to_sql! {
    Boolean(bool),
    TinyInt(i8),
    SmallInt(i16),
    Int(i32),
    BigInt(i64),
    HugeInt(i128),
    UTinyInt(u8),
    USmallInt(u16),
    UInt(u32),
    UBigInt(u64),
    Float(f32),
    Double(f64),
}

// -----------------------------------------------------------------------------

struct StraightToSql<'a>(ValueRef<'a>);

impl duckdb::ToSql for StraightToSql<'_> {
    fn to_sql(&self) -> duckdb::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Borrowed(self.0))
    }
}

// -----------------------------------------------------------------------------

pub trait Params {
    fn bind(self, s: &mut duckdb::Statement) -> duckdb::Result<()>;
}

impl Params for () {
    fn bind(self, _: &mut duckdb::Statement) -> duckdb::Result<()> {
        Ok(())
    }
}

impl<T: ToSql> Params for T {
    fn bind(self, s: &mut duckdb::Statement) -> duckdb::Result<()> {
        s.raw_bind_parameter(1, StraightToSql(self.to_sql()))
    }
}

macro_rules! gen_tuple_params {
    { $( ($($x:ident),* $(,)?) )+ } => { $(
        #[allow(non_snake_case)]
        impl<$($x: ToSql),*> Params for ($($x,)*) {            
            fn bind(self, s: &mut duckdb::Statement) -> duckdb::Result<()> {
                let ($($x,)*) = self;
                let mut c = 0;
                $(
                    c += 1;
                    s.raw_bind_parameter(c, StraightToSql($x.to_sql()))?;
                )*
                Ok(())
            }
        }
    )+ };
}

gen_tuple_params! {
    (A,)
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E)
    (A, B, C, D, E, F)
    (A, B, C, D, E, F, G)
    (A, B, C, D, E, F, G, H)
    (A, B, C, D, E, F, G, H, I)
    (A, B, C, D, E, F, G, H, I, J)
    (A, B, C, D, E, F, G, H, I, J, K)
    (A, B, C, D, E, F, G, H, I, J, K, L)
    (A, B, C, D, E, F, G, H, I, J, K, L, M)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
}
