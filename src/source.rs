use std::collections::VecDeque;

use arrow::array::*;
use arrow::buffer::NullBuffer;
use arrow::datatypes::*;
use half::f16;

#[derive(Debug)]
pub enum Error {
    ContainsNulls,
    WrongArrowType,
    WrongStructWidth,
    ConversionError(Box<dyn std::error::Error>),
}

pub struct Accessor<C: Columns> {
    len: usize,
    source: StructSource<C>,
}

impl<C: Columns> Accessor<C> {
    pub fn from_struct_array(array: StructArray) -> Result<Self, Error> {
        Ok(Accessor {
            len: array.len(),
            source: StructSource::new(array)?,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, i: usize) -> Option<Result<C, Error>> {
        if i < self.len() {
            Some(unsafe { self.source.get_unchecked(i) })
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked(&self, i: usize) -> Result<C, Error> {
        self.source.get_unchecked(i)
    }
}

impl<C: Columns> IntoIterator for Accessor<C> {
    type Item = Result<C, Error>;
    type IntoIter = Iter<C>;

    fn into_iter(self) -> Iter<C> {
        Iter::new(self)
    }
}

// -----------------------------------------------------------------------------

pub struct Iter<C: Columns> {
    i: usize,
    accessor: Accessor<C>,
}

impl<C: Columns> Iter<C> {
    pub fn new(accessor: Accessor<C>) -> Self {
        Iter { i: 0, accessor }
    }
}

impl<C: Columns> Iterator for Iter<C> {
    type Item = Result<C, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.accessor.len() {
            let x = unsafe { self.accessor.get_unchecked(self.i) };
            self.i += 1;
            Some(x)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.len();
        (size, Some(size))
    }
}

impl<C: Columns> ExactSizeIterator for Iter<C> {
    fn len(&self) -> usize {
        self.accessor.len() - self.i
    }
}

// -----------------------------------------------------------------------------

pub trait FromArrow: Sized + 'static {
    type Source;
    fn from_arrow(array: ArrayRef) -> Result<Self::Source, Error>;
    unsafe fn get_unchecked(s: &Self::Source, i: usize) -> Result<Self, Error>;
}

impl<T: Sourced> FromArrow for T
where
    for<'a> <T::Source as Source>::Element<'a>: TryInto<T>,
    for<'a> <<T::Source as Source>::Element<'a> as TryInto<T>>::Error: std::error::Error + 'static,
{
    type Source = T::Source;

    fn from_arrow(array: ArrayRef) -> Result<Self::Source, Error> {
        if array.null_count() != 0 {
            Err(Error::ContainsNulls)
        } else {
            T::Source::from_arrow(array)
        }
    }

    unsafe fn get_unchecked(s: &Self::Source, i: usize) -> Result<T, Error> {
        s.get_unchecked(i)?.try_into()
            .map_err(|e| Error::ConversionError(Box::from(e)))
    }
}

impl<T: FromArrow> FromArrow for Option<T> {
    type Source = NullableSource<T>;

    fn from_arrow(array: ArrayRef) -> Result<Self::Source, Error> {
        Self::Source::from_arrow(array)
    }

    unsafe fn get_unchecked(s: &Self::Source, i: usize) -> Result<Self, Error> {
        Self::Source::get_unchecked(s, i)
    }
}

// -----------------------------------------------------------------------------

pub trait Source: Sized + 'static {
    type Element<'a>;
    fn from_arrow(array: ArrayRef) -> Result<Self, Error>;
    unsafe fn get_unchecked(&self, i: usize) -> Result<Self::Element<'_>, Error>;
}

pub struct BooleanSource(BooleanArray);

impl Source for BooleanSource {
    type Element<'a> = bool;

    fn from_arrow(array: ArrayRef) -> Result<Self, Error> {
        Ok(BooleanSource(
            array.as_any().downcast_ref()
                .map(|r: &BooleanArray| r.clone())
                .ok_or(Error::WrongArrowType)?
        ))
    }

    unsafe fn get_unchecked(&self, i: usize) -> Result<bool, Error> {
        Ok(self.0.value(i))
    }
}

pub struct PrimitiveSource<T: ArrowPrimitiveType>(PrimitiveArray<T>);

impl<T: ArrowPrimitiveType> Source for PrimitiveSource<T> {
    type Element<'a> = T::Native;

    fn from_arrow(array: ArrayRef) -> Result<Self, Error> {
        Ok(PrimitiveSource(
            array.as_any().downcast_ref()
                .map(|r: &PrimitiveArray<T>| r.clone())
                .ok_or(Error::WrongArrowType)?
        ))
    }

    unsafe fn get_unchecked(&self, i: usize) -> Result<T::Native, Error> {
        Ok(self.0.value(i))
    }
}

pub enum StringSource {
    Normal(StringArray),
    Large(LargeStringArray),    
}

impl Source for StringSource {
    type Element<'a> = &'a str;

    fn from_arrow(array: ArrayRef) -> Result<Self, Error> {
        let any = array.as_any();

        if let Some(normal) = any.downcast_ref::<StringArray>() {
            return Ok(Self::Normal(normal.clone()));            
        }

        if let Some(large) = any.downcast_ref::<LargeStringArray>() {
            return Ok(Self::Large(large.clone()));
        }

        Err(Error::WrongArrowType)
    }

    unsafe fn get_unchecked(&self, i: usize) -> Result<Self::Element<'_>, Error> {
        Ok(match self {
            Self::Normal(n) => n.value(i),
            Self::Large(l) => l.value(i),
        })
    }
}

pub enum ByteSource {
    Normal(BinaryArray),
    Large(LargeBinaryArray),
}

impl Source for ByteSource {
    type Element<'a> = &'a [u8];

    fn from_arrow(array: ArrayRef) -> Result<Self, Error> {
        let any = array.as_any();

        if let Some(normal) = any.downcast_ref::<BinaryArray>() {
            return Ok(Self::Normal(normal.clone()));            
        }

        if let Some(large) = any.downcast_ref::<LargeBinaryArray>() {
            return Ok(Self::Large(large.clone()));
        }

        Err(Error::WrongArrowType)
    }

    unsafe fn get_unchecked(&self, i: usize) -> Result<Self::Element<'_>, Error> {
        Ok(match self {
            Self::Normal(n) => n.value(i),
            Self::Large(l) => l.value(i),
        })
    }
}

pub enum NullableSource<T: FromArrow> {
    Normal(T::Source, Option<NullBuffer>),
    Null,
}

impl<T: FromArrow> Source for NullableSource<T> {
    type Element<'a> = Option<T>;

    fn from_arrow(array: ArrayRef) -> Result<Self, Error> {
        if array.as_any().downcast_ref::<NullArray>().is_some() {
            return Ok(Self::Null);
        }

        let nulls = array.nulls().map(|b| b.clone());
        Ok(Self::Normal(T::from_arrow(array)?, nulls))
    }

    unsafe fn get_unchecked(&self, i: usize) -> Result<Option<T>, Error> {
        Ok(match self {
            Self::Null => None,
            Self::Normal(_, Some(nulls)) if nulls.is_null(i) => None,
            Self::Normal(s, _) => Some(T::get_unchecked(s, i)?),
        })
    }
}

pub struct StructSource<C: Columns>(C::Sources);

impl<C: Columns> StructSource<C> {
    fn new(array: StructArray) -> Result<Self, Error> {
        let (_, columns, _) = array.into_parts();
        Ok(StructSource(C::from_arrow(columns.into())?))
    }
}

impl<C: Columns> Source for StructSource<C> {
    type Element<'a> = C;

    fn from_arrow(array: ArrayRef) -> Result<Self, Error> {
        array.as_any().downcast_ref()
            .map(|r: &StructArray| r.clone())
            .ok_or(Error::WrongArrowType)
            .and_then(StructSource::new)
    }

    unsafe fn get_unchecked(&self, i: usize) -> Result<Self::Element<'_>, Error> {
        C::get_unchecked(&self.0, i)
    }
}

// -----------------------------------------------------------------------------

pub trait Sourced: Sized + 'static
where
    for<'a> <Self::Source as Source>::Element<'a>: TryInto<Self>,
    for<'a> <<Self::Source as Source>::Element<'a> as TryInto<Self>>::Error: std::error::Error + 'static,
{
    type Source: Source;
}

pub struct Date<T>(pub T);

impl<T> From<T> for Date<T> {
    fn from(value: T) -> Self {
        Date(value)
    }
}

macro_rules! impl_sourced_primitive {
    { $($native:ty:$arrow:ty)+ } => {
        $(impl Sourced for $native { type Source = PrimitiveSource<$arrow>; })+
    };
}

impl_sourced_primitive! {
    i8:Int8Type  i16:Int16Type  i32:Int32Type  i64:Int64Type
    u8:UInt8Type u16:UInt16Type u32:UInt32Type u64:UInt64Type

    f16:Float16Type
    f32:Float32Type
    f64:Float64Type

    Date<i32>:Date32Type
    Date<i64>:Date64Type
}

macro_rules! impl_sourced {
    { $($source:ty { $($x:ty)+ })+ } => {
        $($(impl Sourced for $x { type Source = $source; })+)+
    };
}

impl_sourced! {
    StringSource { Box<str> String }
    ByteSource { Box<[u8]> Vec<u8> }
}

// -----------------------------------------------------------------------------

pub trait TimeUnit: 'static {
    type Timestamp: ArrowPrimitiveType;
    type Time: ArrowPrimitiveType;
    type Duration: ArrowPrimitiveType;
}

pub struct Second;
pub struct Millisecond;
pub struct Microsecond;
pub struct Nanosecond;

impl TimeUnit for Second {
    type Timestamp = TimestampSecondType;
    type Time = Time32SecondType;
    type Duration = DurationSecondType;
}

impl TimeUnit for Millisecond {
    type Timestamp = TimestampMillisecondType;
    type Time = Time32MillisecondType;
    type Duration = DurationMillisecondType;
}

impl TimeUnit for Microsecond {
    type Timestamp = TimestampMicrosecondType;
    type Time = Time64MicrosecondType;
    type Duration = DurationMicrosecondType;
}

impl TimeUnit for Nanosecond {
    type Timestamp = TimestampNanosecondType;
    type Time = Time64NanosecondType;
    type Duration = DurationNanosecondType;
}

pub trait IntervalUnit: 'static {
    type Interval: ArrowPrimitiveType;
}

pub struct YearMonth;
pub struct DayTime;
pub struct MonthDayNano;

impl IntervalUnit for YearMonth { type Interval = IntervalYearMonthType; }
impl IntervalUnit for DayTime { type Interval = IntervalDayTimeType; }
impl IntervalUnit for MonthDayNano { type Interval = IntervalMonthDayNanoType; }

pub struct Timestamp<T: TimeUnit>(<T::Timestamp as ArrowPrimitiveType>::Native);
pub struct Time<T: TimeUnit>(<T::Time as ArrowPrimitiveType>::Native);
pub struct Duration<T: TimeUnit>(<T::Duration as ArrowPrimitiveType>::Native);
pub struct Interval<T: IntervalUnit>(<T::Interval as ArrowPrimitiveType>::Native);

impl<N, T> From<N> for Timestamp<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Timestamp: ArrowPrimitiveType<Native=N>
{
    fn from(value: N) -> Self {
        Timestamp(value)
    }
}

impl<N, T> From<N> for Time<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Time: ArrowPrimitiveType<Native=N>
{
    fn from(value: N) -> Self {
        Time(value)
    }
}

impl<N, T> From<N> for Duration<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Duration: ArrowPrimitiveType<Native=N>
{
    fn from(value: N) -> Self {
        Duration(value)
    }
}

impl<N, T> From<N> for Interval<T>
where
    N: ArrowNativeType,
    T: IntervalUnit,
    T::Interval: ArrowPrimitiveType<Native=N>
{
    fn from(value: N) -> Self {
        Interval(value)
    }
}

impl<N, T> Sourced for Timestamp<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Timestamp: ArrowPrimitiveType<Native=N>
{
    type Source = PrimitiveSource<T::Timestamp>;
}

impl<N, T> Sourced for Time<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Time: ArrowPrimitiveType<Native=N>
{
    type Source = PrimitiveSource<T::Time>;
}

impl<N, T> Sourced for Duration<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Duration: ArrowPrimitiveType<Native=N>
{
    type Source = PrimitiveSource<T::Duration>;
}

impl<N, T> Sourced for Interval<T>
where
    N: ArrowNativeType,
    T: IntervalUnit,
    T::Interval: ArrowPrimitiveType<Native=N>
{
    type Source = PrimitiveSource<T::Interval>;
}

// -----------------------------------------------------------------------------

pub trait Columns: Sized + 'static {
    type Sources;

    fn from_arrow(columns: Box<[ArrayRef]>) -> Result<Self::Sources, Error>;
    unsafe fn get_unchecked(sources: &Self::Sources, i: usize) -> Result<Self, Error>;
}

impl<T: FromArrow> Columns for T {
    type Sources = T::Source;

    fn from_arrow(columns: Box<[ArrayRef]>) -> Result<Self::Sources, Error> {
        if columns.len() != 1 {
            Err(Error::WrongStructWidth)
        } else {
            let mut columns = Vec::from(columns);
            Ok(T::from_arrow(columns.pop().unwrap())?)
        }
    }

    unsafe fn get_unchecked(sources: &Self::Sources, i: usize) -> Result<T, Error> {
        T::get_unchecked(sources, i)
    }
}

impl Columns for () {
    type Sources = ();

    fn from_arrow(_: Box<[ArrayRef]>) -> Result<Self::Sources, Error> {
        Ok(())
    }

    unsafe fn get_unchecked(_: &Self::Sources, _: usize) -> Result<Self, Error> {
        Ok(())
    }
}

macro_rules! impl_columns {
    { $($n:expr => ($($x:ident),+$(,)?))+ } => { $(
        #[allow(non_snake_case)]
        impl<$($x: FromArrow),+> Columns for ($($x,)+) {
            type Sources = ($($x::Source,)+);

            fn from_arrow(columns: Box<[ArrayRef]>) -> Result<Self::Sources, Error> {
                if columns.len() != $n {
                    Err(Error::WrongStructWidth)
                } else {
                    let mut columns = VecDeque::from(Vec::from(columns));
                    Ok(($($x::from_arrow(columns.pop_front().unwrap())?,)+))
                }
            }

            unsafe fn get_unchecked(sources: &Self::Sources, i: usize) -> Result<Self, Error> {
                let ($($x,)*) = sources;
                Ok(($($x::get_unchecked($x, i)?,)+))
            }
        }
    )+ };
}

impl_columns! {
    1 => (A,)
    2 => (A, B)
    3 => (A, B, C)
    4 => (A, B, C, D)
    5 => (A, B, C, D, E)
    6 => (A, B, C, D, E, F)
    7 => (A, B, C, D, E, F, G)
    8 => (A, B, C, D, E, F, G, H)
    9 => (A, B, C, D, E, F, G, H, I)
    10 => (A, B, C, D, E, F, G, H, I, J)
    11 => (A, B, C, D, E, F, G, H, I, J, K)
    12 => (A, B, C, D, E, F, G, H, I, J, K, L)
    13 => (A, B, C, D, E, F, G, H, I, J, K, L, M)
    14 => (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
    15 => (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
}
