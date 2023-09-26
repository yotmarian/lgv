use std::any;
use std::collections::VecDeque;
use std::marker::PhantomData;

use arrow::array::*;
use arrow::buffer::NullBuffer;
use arrow::datatypes::*;
use arrow::error::{Result, ArrowError};
use half::f16;

pub struct Accessor<T: FromArrow>(T::Accessor);

impl<T: FromArrow> Accessor<T> {
    pub fn from_array(array: T::Array) -> Result<Self> {
        Ok(Accessor(T::from_array(array)?))
    }

    pub fn from_struct(array: StructArray) -> Result<Self> {
        Ok(Accessor(T::from_struct(array)?))
    }

    pub fn from_any(array: ArrayRef) -> Result<Self> {
        Ok(Accessor(T::from_any(array)?))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, i: usize) -> Option<T> {
        if i < self.len() {
            Some(unsafe { self.0.get_unchecked(i) })
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked(&self, i: usize) -> T {
        self.0.get_unchecked(i)
    }
}

impl<T: FromArrow> IntoIterator for Accessor<T> {
    type Item = T;
    type IntoIter = Iter<T>;

    fn into_iter(self) -> Iter<T> {
        Iter::new(self.0)
    }
}

// -----------------------------------------------------------------------------

pub struct Iter<T: FromArrow> {
    i: usize,
    accessor: T::Accessor,
}

impl<T: FromArrow> Iter<T> {
    pub fn new(accessor: T::Accessor) -> Self {
        Iter { i: 0, accessor }
    }
}

impl<T: FromArrow> Iterator for Iter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
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

impl<T: FromArrow> ExactSizeIterator for Iter<T> {
    fn len(&self) -> usize {
        self.accessor.len() - self.i
    }
}

// -----------------------------------------------------------------------------

pub trait InternalAccessor {
    type Item;
    fn len(&self) -> usize;
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item;
}

pub trait FromArrow: Sized {
    type Array: Array + Clone + 'static;
    type Accessor: InternalAccessor<Item = Self>;
    const STRUCT: bool;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Accessor>;

    fn from_array(array: Self::Array) -> Result<Self::Accessor> {
        null_check::<Self, _>(&array)?;
        Self::from_nonnull_array(array)
    }

    fn from_nonnull_struct(array: StructArray) -> Result<Self::Accessor> {
        let mut columns = n_columns::<Self>(1, array)?;
        Self::from_any(columns.pop().unwrap())
    }

    fn from_struct(array: StructArray) -> Result<Self::Accessor> {
        null_check::<Self, _>(&array)?;
        Self::from_nonnull_struct(array)
    }

    fn from_nonnull_any(array: ArrayRef) -> Result<Self::Accessor> {
        Self::from_nonnull_array(downcast::<Self>(array)?)
    }

    fn from_any(array: ArrayRef) -> Result<Self::Accessor> {
        null_check::<Self, _>(&array)?;
        Self::from_nonnull_any(array)
    }
}

impl FromArrow for () {
    type Array = ArrayRef;
    type Accessor = UnitAccessor;
    const STRUCT: bool = false;

    fn from_nonnull_array(array: ArrayRef) -> Result<UnitAccessor> {
        Ok(UnitAccessor::new(array))
    }

    fn from_nonnull_struct(array: StructArray) -> Result<UnitAccessor> {
        Ok(UnitAccessor::new(array))
    }

    fn from_nonnull_any(array: ArrayRef) -> Result<UnitAccessor> {
        Ok(UnitAccessor::new(array))
    }
}

impl<T: DirectAccess> FromArrow for T
where
    for<'a> &'a T::Array: ArrayAccessor,
{
    type Array = T::Array;
    type Accessor = DirectAccessor<T::Array, T>;
    const STRUCT: bool = false;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Accessor> {
        Ok(DirectAccessor::new(array))
    }
}

impl<T: FromArrow> FromArrow for Option<T> {
    type Array = T::Array;
    type Accessor = MaskedAccessor<T::Accessor>;
    const STRUCT: bool = T::STRUCT;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Accessor> {
        Ok(MaskedAccessor::new(T::from_nonnull_array(array)?, None))
    }

    fn from_array(array: Self::Array) -> Result<Self::Accessor> {
        let nulls = clone_nulls(&array);
        Ok(MaskedAccessor::new(T::from_nonnull_array(array)?, nulls))
    }

    fn from_nonnull_struct(array: StructArray) -> Result<Self::Accessor> {
        if T::STRUCT || array.num_columns() > 1 {
            Ok(MaskedAccessor::new(T::from_nonnull_struct(array)?, None))
        } else {
            Self::from_any(n_columns::<Self>(1, array)?.pop().unwrap())
        }
    }

    fn from_struct(array: StructArray) -> Result<Self::Accessor> {
        let nulls = clone_nulls(&array);

        if T::STRUCT || array.num_columns() > 1 {
            Ok(MaskedAccessor::new(T::from_nonnull_struct(array)?, nulls))
        } else {
            let column = n_columns::<Self>(1, array)?.pop().unwrap();
            let inner = Self::from_any(column)?;
            Ok(inner.union_nulls(nulls))
        }
    }

    fn from_nonnull_any(array: ArrayRef) -> Result<Self::Accessor> {
        Self::from_any(array)
    }

    fn from_any(array: ArrayRef) -> Result<Self::Accessor> {
        if let Some(nulls) = array.as_any().downcast_ref::<NullArray>() {
            return Ok(MaskedAccessor::from_nulls(nulls));
        }

        let nulls = clone_nulls(&array);
        Ok(MaskedAccessor::new(T::from_nonnull_any(array)?, nulls))
    }
}

macro_rules! impl_from_arrow {
    { $($n:expr => $ta:ident ($($x:ident),+$(,)?))+ } => { $(
        pub struct $ta<$($x: FromArrow),+>(usize, $($x::Accessor),+);

        impl<$($x: FromArrow),+> InternalAccessor for $ta<$($x),+> {
            type Item = ($($x,)+);

            fn len(&self) -> usize {
                self.0
            }

            #[allow(non_snake_case)]
            unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
                let $ta(_, $($x),+) = self;
                ($($x.get_unchecked(i),)+)
            }
        }

        impl<$($x: FromArrow),+> FromArrow for ($($x,)+) {
            type Array = StructArray;
            type Accessor = $ta<$($x),+>;
            const STRUCT: bool = true;

            fn from_nonnull_array(array: StructArray) -> Result<Self::Accessor> {
                let len = array.len();
                let mut columns = VecDeque::from(n_columns::<Self>($n, array)?);
                Ok($ta(len, $(
                    $x::from_any(columns.pop_front().unwrap())?
                ),+))
            }
        }
    )+ };
}

impl_from_arrow! {
    1 => TA1 (A,)
    2 => TA2 (A, B)
    3 => TA3 (A, B, C)
    4 => TA4 (A, B, C, D)
    5 => TA5 (A, B, C, D, E)
    6 => TA6 (A, B, C, D, E, F)
    7 => TA7 (A, B, C, D, E, F, G)
    8 => TA8 (A, B, C, D, E, F, G, H)
    9 => TA9 (A, B, C, D, E, F, G, H, I)
    10 => TA10 (A, B, C, D, E, F, G, H, I, J)
    11 => TA11 (A, B, C, D, E, F, G, H, I, J, K)
    12 => TA12 (A, B, C, D, E, F, G, H, I, J, K, L)
    13 => TA13 (A, B, C, D, E, F, G, H, I, J, K, L, M)
    14 => TA14 (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
    15 => TA15 (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
}

// -----------------------------------------------------------------------------

pub trait DirectAccess
where
    Self::Array: Array + Clone + 'static,
    for<'a> &'a Self::Array: ArrayAccessor,
    for<'a> Self: From<<&'a Self::Array as ArrayAccessor>::Item>,
{
    type Array;
}

impl DirectAccess for bool { type Array = BooleanArray; }

macro_rules! impl_has_accessor {
    { $($array:ident { $($native:ty:$arrow:ty)+ })+ } => {
        $($(impl DirectAccess for $native { type Array = $array<$arrow>; })+)+
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

impl<T: ArrowPrimitiveType> DirectAccess for Primitive<T> {
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

impl<'a, T, U> DirectAccess for Bytes<T, U>
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

impl<'a, T> DirectAccess for FixedSizeBytes<T>
where
    for<'b> T: From<&'b [u8]>,
{
    type Array = FixedSizeBinaryArray;
}

// -----------------------------------------------------------------------------

pub struct List<T: FromArrow, O = i32> {
    pub value: Result<Accessor<T>>,
    phantom: PhantomData<O>,
}

impl<T: FromArrow, O> List<T, O> {
    pub fn new(value: Result<Accessor<T>>) -> Self {
        List { value, phantom: PhantomData }
    }
}

impl<T: FromArrow, O> From<ArrayRef> for List<T, O> {
    fn from(value: ArrayRef) -> Self {
        Self::new(Accessor::from_any(value))
    }
}

impl<T: FromArrow, O: OffsetSizeTrait> DirectAccess for List<T, O> {
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

impl<N, T> DirectAccess for Timestamp<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Timestamp: ArrowPrimitiveType<Native=N>
{
    type Array = PrimitiveArray<T::Timestamp>;
}

impl<N, T> DirectAccess for Time<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Time: ArrowPrimitiveType<Native=N>
{
    type Array = PrimitiveArray<T::Time>;
}

impl<N, T> DirectAccess for Duration<T>
where
    N: ArrowNativeType,
    T: TimeUnit,
    T::Duration: ArrowPrimitiveType<Native=N>
{
    type Array = PrimitiveArray<T::Duration>;
}

impl<N, T> DirectAccess for Interval<T>
where
    N: ArrowNativeType,
    T: IntervalUnit,
    T::Interval: ArrowPrimitiveType<Native=N>
{
    type Array = PrimitiveArray<T::Interval>;
}

// -----------------------------------------------------------------------------

pub struct UnitAccessor(usize);

impl UnitAccessor {
    fn new<A: Array>(array: A) -> UnitAccessor {
        UnitAccessor(array.len())
    }
}

impl InternalAccessor for UnitAccessor {
    type Item = ();
    fn len(&self) -> usize { self.0 }
    unsafe fn get_unchecked(&self, _: usize) -> () { }
}

// -----------------------------------------------------------------------------

pub struct DirectAccessor<A, T> {
    array: A,
    phantom: PhantomData<T>,
}

impl<A, T> DirectAccessor<A, T> {
    fn new(array: A) -> DirectAccessor<A, T> {
        DirectAccessor { array, phantom: PhantomData }
    }
}

impl<A, T> InternalAccessor for DirectAccessor<A, T>
where
    A: Array,
    T: DirectAccess<Array=A>,
    for<'a> &'a A: ArrayAccessor,
{
    type Item = T;

    fn len(&self) -> usize {
        Array::len(&self.array)
    }

    unsafe fn get_unchecked(&self, i: usize) -> T {
        T::from(ArrayAccessor::value(&&self.array, i))
    }
}

// -----------------------------------------------------------------------------

pub struct MaskedAccessor<A>(
    std::result::Result<(A, Option<NullBuffer>), usize>,
);

impl<A> MaskedAccessor<A> {
    fn new(accessor: A, nulls: Option<NullBuffer>) -> Self {
        MaskedAccessor(Ok((accessor, nulls)))
    }

    fn from_nulls(nulls: &NullArray) -> Self {
        MaskedAccessor(Err(nulls.len()))
    }

    fn union_nulls(self, nulls: Option<NullBuffer>) -> Self {
        MaskedAccessor(match self.0 {
            Err(count) => Err(count),
            Ok((a, n)) => Ok((a, match (n, nulls) {
                (Some(lhs), Some(rhs)) => Some(NullBuffer::new(lhs.inner() & rhs.inner())),
                (None, b) => b,
                (b, None) => b,
            }))
        })
    }
}

impl<A: InternalAccessor> InternalAccessor for MaskedAccessor<A> {
    type Item = Option<A::Item>;

    fn len(&self) -> usize {
        match &self.0 {
            Ok((accesor, _)) => accesor.len(),
            Err(count) => *count,
        }
    }

    unsafe fn get_unchecked(&self, i: usize) -> Self::Item {
        let (accesor, nulls) = self.0.as_ref().ok()?;

        if null_at(nulls, i) {
            return None;
        }

        Some(accesor.get_unchecked(i))
    }
}

// -----------------------------------------------------------------------------

fn null_at(nulls: &Option<NullBuffer>, i: usize) -> bool {
    nulls.as_ref().map_or(false, |b| b.is_null(i))
}

fn clone_nulls<A: Array>(array: &A) -> Option<NullBuffer> {
    array.nulls().map(|b| b.clone())
}

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
