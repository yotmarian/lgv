use std::borrow::Cow;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use duckdb::{ToSql, Connection};
use duckdb::types::{Value, Null};
use uuid::Uuid;

use crate::arrow::FromArrow;

pub type DuckNull = Null;
pub type DuckValue = Value;
pub type DuckError = duckdb::Error;
pub type DuckStatement<'conn> = duckdb::Statement<'conn>;
pub type DuckResult<T> = duckdb::Result<T>;

pub fn has_table(conn: &Connection) -> Result<QueryRow<&str, bool>> {
    let sql = "
        select count(*) == 1
        from information_schema.tables
        where table_name = ?
    ";

    QueryRow::new(conn, sql)
}

// -----------------------------------------------------------------------------

pub struct Statement<'conn, P: Params> {
    duck: DuckStatement<'conn>,
    sql: &'static str,
    phantom: PhantomData<fn(P)>,
}

impl<'conn, P: Params> Statement<'conn, P> {
    pub fn new(conn: &'conn Connection, sql: &'static str) -> Result<Statement<'conn, P>> {
        let duck = conn.prepare(sql).context(sql)?;
        duck.column_count();
        Ok(Statement { duck, sql, phantom: PhantomData })
    }

    pub fn execute<'a>(&'a mut self, params: P) -> Result<usize> {
        params.bind(&mut self.duck)
            .and_then(|()| self.duck.raw_execute())
            .context(self.sql)
    }
}

// -----------------------------------------------------------------------------

pub struct Query<'conn, P: Params, R: FromArrow> {
    stmt: Statement<'conn, P>,
    phantom: PhantomData<fn(P) -> R>,
}

impl<'conn, P: Params, R: FromArrow> Query<'conn, P, R> {
    pub fn new(conn: &'conn Connection, sql: &'static str) -> Result<Query<'conn, P, R>> {
        Ok(Query { stmt: Statement::new(conn, sql)?, phantom: PhantomData })
    }

    pub fn execute(&mut self, params: P) -> Result<QueryResults<'_, 'conn, R>> {
        self.stmt.execute(params)?;

        Ok(QueryResults {
            statement: &mut self.stmt.duck,
            rows: None,
            sql: self.stmt.sql,
        })
    }
}

// -----------------------------------------------------------------------------

pub struct QueryRow<'conn, P: Params, R: FromArrow>(Query<'conn, P, R>);

impl<'conn, P: Params, R: FromArrow> QueryRow<'conn, P, R> {
    pub fn new(conn: &'conn Connection, sql: &'static str) -> Result<QueryRow<'conn, P, R>> {
        Ok(QueryRow(Query::new(conn, sql)?))
    }

    pub fn execute(&mut self, params: P) -> Result<R> {
        match self.0.execute(params)?.next() {
            Some(row) => row,
            None => Err(DuckError::QueryReturnedNoRows).context(self.0.stmt.sql),
        }
    }
}

// -----------------------------------------------------------------------------

pub struct QueryResults<'a, 'conn, R: FromArrow> {
    statement: &'a mut DuckStatement<'conn>,
    rows: Option<R::Iter>,
    sql: &'static str,
}

impl<'a, 'conn, R: FromArrow> Iterator for QueryResults<'a, 'conn, R> {
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

            match R::from_struct(self.statement.step()?) {
                Ok(rows) => self.rows = Some(rows),
                Err(e) => return Some(Err(e).context(self.sql))
            }
        }
    }
}

// -----------------------------------------------------------------------------

pub trait Params {
    fn bind<'conn>(&self, s: &mut DuckStatement<'conn>) -> DuckResult<()>;
}

impl<T: ToSql + Marker> Params for T {
    fn bind<'conn>(&self, s: &mut DuckStatement<'conn>) -> DuckResult<()> {
        s.raw_bind_parameter(1, &self.to_sql()?)
    }
}

trait Marker: ToSql { }
impl<T: Marker> Marker for Option<T> { }
impl<T: Marker + ?Sized> Marker for &'_ T { }
impl<T: Marker + ?Sized> Marker for Box<T> { }
impl<T: Marker + ?Sized> Marker for Rc<T> { }
impl<T: Marker + ?Sized> Marker for Arc<T> { }
impl<T: Marker + ToOwned + ?Sized> Marker for Cow<'_, T> { }

macro_rules! mark {
    {$($x:ty)+} => { $(impl Marker for $x { })+ };
}

mark! {
    Null bool String str Vec<u8> [u8] Value Duration Uuid
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 usize f32 f64
}

macro_rules! gen_tuple_params {
    { $(($($x:ident)*))+ } => { $(
        #[allow(non_snake_case, unused_variables, unused_mut, unused_assignments)]
        impl<$($x: ToSql),*> Params for ($($x,)*) {            
            fn bind(&self, s: &mut DuckStatement) -> DuckResult<()> {
                let ($($x,)*) = self;
                let mut c = 1;
                $(
                    s.raw_bind_parameter(c, &$x.to_sql()?)?;
                    c += 1;
                )*
                Ok(())
            }
        }
    )+ };
}

gen_tuple_params! {
    ()
    (A)
    (A B)
    (A B C)
    (A B C D)
    (A B C D E)
    (A B C D E F)
    (A B C D E F G)
    (A B C D E F G H)
    (A B C D E F G H I)
    (A B C D E F G H I J)
    (A B C D E F G H I J K)
    (A B C D E F G H I J K L)
    (A B C D E F G H I J K L M)
    (A B C D E F G H I J K L M N)
    (A B C D E F G H I J K L M N O)
}
