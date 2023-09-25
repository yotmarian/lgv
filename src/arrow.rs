use std::any;
use std::collections::VecDeque;
use std::iter::{Once, once, Repeat, Take, repeat};
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::*;
use arrow::buffer::NullBuffer;
use arrow::datatypes::*;
use arrow::error::{Result, ArrowError};
use half::f16;

pub trait FromArrow: Sized {
    type Array: Array + Clone + 'static;
    type Iter: Iterator<Item = Self>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter>;

    fn from_array(array: Self::Array) -> Result<Self::Iter> {
        null_check::<Self, _>(&array)?;
        Self::from_nonnull_array(array)
    }

    fn from_nonnull_struct(array: StructArray) -> Result<Self::Iter> {
        let mut columns = n_columns::<Self>(1, array)?;
        Self::from_any(columns.pop().unwrap())
    }

    fn from_struct(array: StructArray) -> Result<Self::Iter> {
        null_check::<Self, _>(&array)?;
        Self::from_nonnull_struct(array)
    }

    fn from_nonnull_any(array: ArrayRef) -> Result<Self::Iter> {
        Self::from_nonnull_array(downcast::<Self>(array)?)
    }

    fn from_any(array: ArrayRef) -> Result<Self::Iter> {
        null_check::<Self, _>(&array)?;
        Self::from_nonnull_any(array)
    }
}

impl FromArrow for () {
    type Array = ArrayRef;
    type Iter = Take<Repeat<()>>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(repeat(()).take(array.len()))
    }

    fn from_nonnull_struct(array: StructArray) -> Result<Self::Iter> {
        Ok(repeat(()).take(array.len()))
    }

    fn from_nonnull_any(array: ArrayRef) -> Result<Self::Iter> {
        Ok(repeat(()).take(array.len()))
    }
}

impl<T: HasAccessor> FromArrow for T
where
    for<'a> &'a T::Array: ArrayAccessor,
{
    type Array = T::Array;
    type Iter = ArrayIter<T::Array, T>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(ArrayIter::new(array))
    }
}

impl<T: FromArrow> FromArrow for Option<T> {
    type Array = T::Array;
    type Iter = Mask<T::Iter>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(Mask::new(T::from_nonnull_array(array)?, None))
    }

    fn from_array(array: Self::Array) -> Result<Self::Iter> {
        let nulls = array.nulls().map(|r| r.clone());
        Ok(Mask::new(T::from_nonnull_array(array)?, nulls))
    }

    fn from_nonnull_struct(array: StructArray) -> Result<Self::Iter> {
        Ok(Mask::new(T::from_nonnull_struct(array)?, None))
    }

    fn from_struct(array: StructArray) -> Result<Self::Iter> {
        let nulls = array.nulls().map(|r| r.clone());
        Ok(Mask::new(T::from_nonnull_struct(array)?, nulls))
    }

    fn from_nonnull_any(array: ArrayRef) -> Result<Self::Iter> {
        Ok(Mask::new(T::from_nonnull_any(array)?, None))
    }

    fn from_any(array: ArrayRef) -> Result<Self::Iter> {
        let nulls = array.nulls().map(|r| r.clone());
        Ok(Mask::new(T::from_nonnull_any(array)?, nulls))
    }
}

impl FromArrow for ArrayRef {
    type Array = ArrayRef;
    type Iter = Once<ArrayRef>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(once(array))
    }

    fn from_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(once(array))
    }

    fn from_nonnull_struct(array: StructArray) -> Result<Self::Iter> {
        Ok(once(Arc::new(array)))
    }

    fn from_struct(array: StructArray) -> Result<Self::Iter> {
        Ok(once(Arc::new(array)))
    }

    fn from_nonnull_any(array: ArrayRef) -> Result<Self::Iter> {
        Ok(once(array))
    }

    fn from_any(array: ArrayRef) -> Result<Self::Iter> {
        Ok(once(array))
    }
}

macro_rules! gen_from_arrow_tuples {
    ($($n:expr => $zip:ident ($($x:ident),+))+) => { $(
        pub struct $zip<$($x),+>($($x),+);

        impl<$($x: Iterator),+> Iterator for $zip<$($x),+> {
            type Item = ($($x::Item),+);

            #[allow(non_snake_case)]
            fn next(&mut self) -> Option<Self::Item> {
                let $zip($($x),+) = self;
                Some(($($x.next()?),+))
            }
        }

        impl<$($x: FromArrow),+> FromArrow for ($($x),+) {
            type Array = StructArray;
            type Iter = $zip<$($x::Iter),+>;

            fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
                let mut columns = VecDeque::from(n_columns::<Self>($n, array)?);
                Ok($zip($($x::from_any(columns.pop_front().unwrap())?),+))
            }
        }
    )+ };
}

gen_from_arrow_tuples!(
    2 => Zip2 (A, B)
    3 => Zip3 (A, B, C)
    4 => Zip4 (A, B, C, D)
    5 => Zip5 (A, B, C, D, E)
    6 => Zip6 (A, B, C, D, E, F)
    7 => Zip7 (A, B, C, D, E, F, G)
    8 => Zip8 (A, B, C, D, E, F, G, H)
    9 => Zip9 (A, B, C, D, E, F, G, H, I)
    10 => Zip10 (A, B, C, D, E, F, G, H, I, J)
    11 => Zip11 (A, B, C, D, E, F, G, H, I, J, K)
    12 => Zip12 (A, B, C, D, E, F, G, H, I, J, K, L)
    13 => Zip13 (A, B, C, D, E, F, G, H, I, J, K, L, M)
    14 => Zip14 (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
    15 => Zip15 (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
);

// -----------------------------------------------------------------------------

pub trait HasAccessor
where
    Self::Array: Array + Clone + 'static,
    for<'a> &'a Self::Array: ArrayAccessor,
    for<'a> Self: From<<&'a Self::Array as ArrayAccessor>::Item>,
{
    type Array;
}

impl HasAccessor for bool { type Array = BooleanArray; }

macro_rules! impl_has_accessor {
    { $($array:ident { $($native:ty:$arrow:ty)+ })+ } => {
        $($(impl HasAccessor for $native { type Array = $array<$arrow>; })+)+
    };
}

impl_has_accessor! {
    PrimitiveArray {
        i8:Int8Type  i16:Int16Type  i32:Int32Type  i64:Int64Type
        u8:UInt8Type u16:UInt16Type u32:UInt32Type u64:UInt64Type

        f16:Float16Type
        f32:Float32Type
        f64:Float64Type

        Date<i32>:Date32Type
        Date<i64>:Date64Type
    }

    GenericByteArray {
        Box<str>:Utf8Type
        String:Utf8Type

        Box<[u8]>:BinaryType
        Vec<u8>:BinaryType

        Large<Box<str>>:LargeUtf8Type
        Large<String>:LargeUtf8Type

        Large<Box<[u8]>>:LargeBinaryType
        Large<Vec<u8>>:LargeBinaryType
    }
}

pub struct Date<T>(pub T);

impl<T> From<T> for Date<T> {
    fn from(value: T) -> Self {
        Date(value)
    }
}

pub struct Large<T>(pub T);

impl<'a, T, S: ?Sized> From<&'a S> for Large<T>
where
    for<'b> T: From<&'b S>,
{
    fn from(value: &'a S) -> Self {
        Large(T::from(value))
    }
}

// -----------------------------------------------------------------------------

pub struct Primitive<T: ArrowPrimitiveType>(pub T::Native);

impl<N, T> From<N> for Primitive<T>
where
    N: ArrowNativeType,
    T: ArrowPrimitiveType<Native=N>,
{
    fn from(value: N) -> Self {
        Primitive(value)
    }
}

impl<T: ArrowPrimitiveType> HasAccessor for Primitive<T> {
    type Array = PrimitiveArray<T>;
}

// -----------------------------------------------------------------------------

pub struct Bytes<T, U> {
    pub value: U,
    phantom: PhantomData<T>,
}

impl<T, U> Bytes<T, U> {
    pub fn new(value: U) -> Bytes<T, U> {
        Bytes { value, phantom: PhantomData }
    }
}

impl<'a, T, U> From<&'a T::Native> for Bytes<T, U>
where
    T: ByteArrayType,
    U: From<&'a T::Native>,
{
    fn from(value: &'a T::Native) -> Bytes<T, U> {
        Bytes::new(U::from(value))
    }
}

impl<'a, T, U> HasAccessor for Bytes<T, U>
where
    T: ByteArrayType,
    for<'b> U: From<&'b T::Native>,
{
    type Array = GenericByteArray<T>;
}

// -----------------------------------------------------------------------------

pub struct FixedSizeBytes<T>(pub T);

impl<'a, T> From<&'a [u8]> for FixedSizeBytes<T>
where
    for<'b> T: From<&'b [u8]>,
{
    fn from(value: &'a [u8]) -> FixedSizeBytes<T> {
        FixedSizeBytes(T::from(value))
    }
}

impl<'a, T> HasAccessor for FixedSizeBytes<T>
where
    for<'b> T: From<&'b [u8]>,
{
    type Array = FixedSizeBinaryArray;
}

// -----------------------------------------------------------------------------

pub struct List<T: FromArrow, O = i32> {
    pub value: Result<T::Iter>,
    phantom: PhantomData<O>,
}

impl<T: FromArrow, O> List<T, O> {
    pub fn new(value: Result<T::Iter>) -> Self {
        List { value, phantom: PhantomData }
    }
}

impl<T: FromArrow, O> From<ArrayRef> for List<T, O> {
    fn from(value: ArrayRef) -> Self {
        Self::new(T::from_any(value))
    }
}

impl<T: FromArrow, O: OffsetSizeTrait> HasAccessor for List<T, O> {
    type Array = GenericListArray<O>;
}

// -----------------------------------------------------------------------------

pub trait TimeUnit {
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

pub trait IntervalUnit {
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

impl<N, T> HasAccessor for Timestamp<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Timestamp: ArrowPrimitiveType<Native=N>
{
    type Array = PrimitiveArray<T::Timestamp>;
}

impl<N, T> HasAccessor for Time<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Time: ArrowPrimitiveType<Native=N>
{
    type Array = PrimitiveArray<T::Time>;
}

impl<N, T> HasAccessor for Duration<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Duration: ArrowPrimitiveType<Native=N>
{
    type Array = PrimitiveArray<T::Duration>;
}

impl<N, T> HasAccessor for Interval<T>
where
    N: ArrowNativeType,
    T: IntervalUnit,
    T::Interval: ArrowPrimitiveType<Native=N>
{
    type Array = PrimitiveArray<T::Interval>;
}

// -----------------------------------------------------------------------------

pub struct ArrayIter<A, T> {
    i: usize,
    array: A,
    phantom: PhantomData<T>,
}

impl<A, T> ArrayIter<A, T> {
    fn new(array: A) -> ArrayIter<A, T> {
        ArrayIter { i: 0, array, phantom: PhantomData }
    }
}

impl<A, T> Iterator for ArrayIter<A, T>
where
    A: Array,
    for<'a> &'a A: ArrayAccessor,
    for<'a> T: From<<&'a A as ArrayAccessor>::Item>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < Array::len(&self.array) {
            let x = ArrayAccessor::value(&&self.array, self.i);
            self.i += 1;
            Some(T::from(x))
        } else {
            None
        }
    }
}

// -----------------------------------------------------------------------------

pub struct Mask<I> {
    iter: I,
    i: usize,
    nulls: Option<NullBuffer>,
}

impl<I> Mask<I> {
    fn new(iter: I, nulls: Option<NullBuffer>) -> Mask<I> {
        Mask { iter, i: 0, nulls }
    }
}

impl<I: Iterator> Iterator for Mask<I> {
    type Item = Option<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next()?;

        Some(match &self.nulls {
            None => {
                Some(item)
            },
            Some(buf) => {
                let x = if buf.is_valid(self.i) {
                    Some(item)
                } else {
                    None
                };

                self.i += 1;
                x
            },
        })
    }
}

// -----------------------------------------------------------------------------

fn null_check<T, A: Array>(array: &A) -> Result<()> {
    if array.null_count() != 0 {
        Err(err_null::<T, A>())
    } else {
        Ok(())
    }
}

fn n_columns<T>(n: usize, array: StructArray) -> Result<Vec<ArrayRef>> {
    let (_, columns, _) = array.into_parts();

    if columns.len() == n {
        Ok(columns)
    } else {
        Err(err_columns::<T>(n, columns.len()))
    }
}

fn downcast<T: FromArrow>(array: ArrayRef) -> Result<T::Array> {
    match array.as_any().downcast_ref::<T::Array>() {
        Some(r) => Ok(r.clone()),
        None => Err(err_downcast::<T, T::Array>())
    }
}

#[cold]
fn err_null<T, A>() -> ArrowError {
    ArrowError::CastError(format!(
        "FromArrow({}): provided {} contains nulls",
        any::type_name::<T>(),
        any::type_name::<A>(),
    ))
}

#[cold]
fn err_columns<T>(required: usize, provided: usize) -> ArrowError {
    ArrowError::CastError(format!(
        "FromArrow({}): provided StructArray has {} columns (required {})",
        any::type_name::<T>(),
        provided,
        required,
    ))
}

#[cold]
fn err_downcast<T, A>() -> ArrowError {
    ArrowError::CastError(format!(
        "FromArrow({}): provided ArrayRef is not a {}",
        any::type_name::<T>(),
        any::type_name::<A>(),
    ))
}
