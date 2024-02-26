use std::{borrow::Cow, marker::PhantomData, rc::Rc, sync::Arc};

use anyhow::{anyhow, Context, Result};
use arrow::array as aa;
pub use duckdb::types::{Null, ValueRef};
use duckdb::{types::ToSqlOutput, Connection};

use crate::arrow::{Array, FromArrow, Iter};

pub trait Spec: Sized {
    type Params<'a>: Params = ();
    type Results: Results = ();
    const SQL: &'static str;

    fn execute(conn: &Connection, params: Self::Params<'_>) -> Result<usize> {
        Statement::<Self>::new(conn)?.execute(params)
    }

    fn select_row(
        conn: &Connection,
        params: Self::Params<'_>,
    ) -> Result<Self::Results> {
        Select::<Self>::new(conn)?.select_row(params)
    }

    fn select_vec(
        conn: &Connection,
        params: Self::Params<'_>,
    ) -> Result<Vec<Self::Results>> {
        Select::<Self>::new(conn)?.select_vec(params)
    }

    fn select_n<const N: usize>(
        conn: &Connection,
        params: Self::Params<'_>,
    ) -> Result<[Self::Results; N]> {
        Select::<Self>::new(conn)?.select_n(params)
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
        Ok(Statement {
            duck,
            phantom: PhantomData,
        })
    }

    pub fn execute(&mut self, params: S::Params<'_>) -> Result<usize> {
        let mut binder = Binder::new(&mut self.duck, S::SQL);
        params.bind_to(&mut binder)?;
        self.duck.raw_execute().context(S::SQL)
    }
}

// -----------------------------------------------------------------------------

pub struct Insert<'conn, S> {
    statement: Statement<'conn, S>,
    conn: &'conn Connection,
}

impl<'conn, S: Spec> Insert<'conn, S> {
    pub fn new(conn: &'conn Connection) -> Result<Insert<'conn, S>> {
        Ok(Insert {
            conn,
            statement: Statement::new(conn)?,
        })
    }

    pub fn insert(&mut self, params: S::Params<'_>) -> Result<usize> {
        self.statement.execute(params)
    }

    pub fn insert_many<'a>(
        &mut self,
        rows: impl IntoIterator<Item = S::Params<'a>>,
    ) -> Result<usize> {
        let mut affected_rows = 0;
        let tx = self.conn.unchecked_transaction()?;

        for params in rows {
            affected_rows += self.statement.execute(params)?;
        }

        tx.commit()?;
        Ok(affected_rows)
    }
}

// -----------------------------------------------------------------------------

pub struct Select<'conn, S>(Statement<'conn, S>);

impl<'conn, S: Spec> Select<'conn, S> {
    pub fn new(conn: &'conn Connection) -> Result<Select<'conn, S>> {
        Ok(Select(Statement::new(conn)?))
    }

    pub fn select(
        &mut self,
        params: S::Params<'_>,
    ) -> Result<QueryResults<'_, 'conn, S::Results>> {
        self.0.execute(params)?;

        Ok(QueryResults {
            statement: &mut self.0.duck,
            rows: None,
            sql: S::SQL,
        })
    }

    pub fn select_row(&mut self, params: S::Params<'_>) -> Result<S::Results> {
        match self.select(params)?.next() {
            Some(row) => row,
            None => Err(duckdb::Error::QueryReturnedNoRows).context(S::SQL),
        }
    }

    pub fn select_vec(
        &mut self,
        params: S::Params<'_>,
    ) -> Result<Vec<S::Results>> {
        self.select(params)?.try_collect()
    }

    pub fn select_n<const N: usize>(
        &mut self,
        params: S::Params<'_>,
    ) -> Result<[S::Results; N]> {
        let mut iter = self.select(params)?;

        std::array::try_from_fn(|i| {
            iter.next()
                .ok_or_else(|| {
                    anyhow!("Not enough rows (expected: {N}, provided: {i})")
                        .context(S::SQL)
                })
                .flatten()
        })
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
    type Iter: Iterator<Item = Self>;

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

pub struct Binder<'a, 'conn> {
    statement: &'a mut duckdb::Statement<'conn>,
    column: usize,
    sql: &'a str,
}

impl<'a, 'conn> Binder<'a, 'conn> {
    fn new(statement: &'a mut duckdb::Statement<'conn>, sql: &'a str) -> Self {
        Binder {
            statement,
            column: 1,
            sql,
        }
    }

    pub fn bind<T: ToSql>(&mut self, value: T) -> Result<()> {
        self.bind_value_ref(value.to_sql())
    }

    pub fn bind_value_ref(&mut self, value_ref: ValueRef) -> Result<()> {
        self.statement
            .raw_bind_parameter(self.column, StraightToSql(value_ref))
            .with_context(|| format!("Column {}", self.column))
            .with_context(|| String::from(self.sql))?;

        self.column += 1;
        Ok(())
    }
}

// -----------------------------------------------------------------------------

pub trait Params {
    fn bind_to(self, binder: &mut Binder) -> Result<()>;
}

impl Params for () {
    fn bind_to(self, _: &mut Binder) -> Result<()> {
        Ok(())
    }
}

impl<T: ToSql> Params for T {
    fn bind_to(self, binder: &mut Binder) -> Result<()> {
        binder.bind(self)
    }
}

macro_rules! gen_tuple_params {
    { $( ($($x:ident),* $(,)?) )+ } => { $(
        #[allow(non_snake_case)]
        impl<$($x: ToSql),*> Params for ($($x,)*) {
            fn bind_to(self, binder: &mut Binder) -> Result<()> {
                let ($($x,)*) = self;
                $(binder.bind($x)?;)*
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
