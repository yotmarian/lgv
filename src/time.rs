use arrow::datatypes::*;
use chrono::{DateTime, TimeZone};
use duckdb::types as dt;

use crate::{
    arrow::Primitive,
    duckdb::{ToSql, ValueRef},
};

pub const MICROSECONDS_IN_SECOND: i64 = 1_000_000;

pub trait IntoTimestamp {
    fn into_timestamp(self) -> Timestamp;
}

impl<Tz: TimeZone> IntoTimestamp for DateTime<Tz> {
    fn into_timestamp(self) -> Timestamp {
        Timestamp(self.timestamp_micros())
    }
}

impl IntoTimestamp for f64 {
    fn into_timestamp(self) -> Timestamp {
        Timestamp((self * MICROSECONDS_IN_SECOND as f64) as i64)
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct Date<T>(pub T);

impl<T> From<T> for Date<T> {
    fn from(value: T) -> Self {
        Date(value)
    }
}

impl Primitive for Date<i32> {
    type Arrow = Date32Type;
}

impl Primitive for Date<i64> {
    type Arrow = Date64Type;
}

impl ToSql for Date<i32> {
    fn to_sql(&self) -> ValueRef {
        ValueRef::Date32(self.0)
    }
}

// -----------------------------------------------------------------------------

pub trait TimeUnit: 'static {
    type Timestamp: ArrowPrimitiveType;
    type Time: ArrowPrimitiveType;
    type Duration: ArrowPrimitiveType;
    const DUCKDB: dt::TimeUnit;
}

pub struct Second;
pub struct Millisecond;
pub struct Microsecond;
pub struct Nanosecond;

impl TimeUnit for Second {
    type Timestamp = TimestampSecondType;
    type Time = Time32SecondType;
    type Duration = DurationSecondType;
    const DUCKDB: dt::TimeUnit = dt::TimeUnit::Second;
}

impl TimeUnit for Millisecond {
    type Timestamp = TimestampMillisecondType;
    type Time = Time32MillisecondType;
    type Duration = DurationMillisecondType;
    const DUCKDB: dt::TimeUnit = dt::TimeUnit::Millisecond;
}

impl TimeUnit for Microsecond {
    type Timestamp = TimestampMicrosecondType;
    type Time = Time64MicrosecondType;
    type Duration = DurationMicrosecondType;
    const DUCKDB: dt::TimeUnit = dt::TimeUnit::Microsecond;
}

impl TimeUnit for Nanosecond {
    type Timestamp = TimestampNanosecondType;
    type Time = Time64NanosecondType;
    type Duration = DurationNanosecondType;
    const DUCKDB: dt::TimeUnit = dt::TimeUnit::Nanosecond;
}

pub trait IntervalUnit: 'static {
    type Interval: ArrowPrimitiveType;
}

pub struct YearMonth;
pub struct DayTime;
pub struct MonthDayNano;

impl IntervalUnit for YearMonth {
    type Interval = IntervalYearMonthType;
}

impl IntervalUnit for DayTime {
    type Interval = IntervalDayTimeType;
}

impl IntervalUnit for MonthDayNano {
    type Interval = IntervalMonthDayNanoType;
}

pub struct Timestamp<T: TimeUnit=Microsecond>(
    pub <T::Timestamp as ArrowPrimitiveType>::Native,
);

pub struct Time<T: TimeUnit=Microsecond>(
    pub <T::Time as ArrowPrimitiveType>::Native,
);

pub struct Duration<T: TimeUnit=Microsecond>(
    pub <T::Duration as ArrowPrimitiveType>::Native,
);

pub struct Interval<T: IntervalUnit=DayTime>(
    pub <T::Interval as ArrowPrimitiveType>::Native,
);

impl<N, T> From<N> for Timestamp<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Timestamp: ArrowPrimitiveType<Native=N>,
{
    fn from(value: N) -> Self {
        Timestamp(value)
    }
}

impl<N, T> From<N> for Time<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Time: ArrowPrimitiveType<Native=N>,
{
    fn from(value: N) -> Self {
        Time(value)
    }
}

impl<N, T> From<N> for Duration<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Duration: ArrowPrimitiveType<Native=N>,
{
    fn from(value: N) -> Self {
        Duration(value)
    }
}

impl<N, T> From<N> for Interval<T>
where
    N: ArrowNativeType,
    T: IntervalUnit,
    T::Interval: ArrowPrimitiveType<Native=N>,
{
    fn from(value: N) -> Self {
        Interval(value)
    }
}

impl<T: TimeUnit> Primitive for Timestamp<T> {
    type Arrow = T::Timestamp;
}

impl<T: TimeUnit> Primitive for Time<T> {
    type Arrow = T::Time;
}

impl<T: TimeUnit> Primitive for Duration<T> {
    type Arrow = T::Duration;
}

impl<T: IntervalUnit> Primitive for Interval<T> {
    type Arrow = T::Interval;
}

impl<T: TimeUnit> ToSql for Timestamp<T>
where
    <T::Timestamp as ArrowPrimitiveType>::Native: Into<i64>,
{
    fn to_sql(&self) -> ValueRef {
        ValueRef::Timestamp(T::DUCKDB, self.0.into())
    }
}

impl<T: TimeUnit> ToSql for Time<T>
where
    <T::Time as ArrowPrimitiveType>::Native: Into<i64>,
{
    fn to_sql(&self) -> ValueRef {
        ValueRef::Time64(T::DUCKDB, self.0.into())
    }
}
