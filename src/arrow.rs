use std::any;
use std::collections::VecDeque;
use std::iter::{Once, once};
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

impl FromArrow for bool {
    type Array = BooleanArray;
    type Iter = ArrayIter<BooleanArray, bool>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(ArrayIter::new(array))
    }
}

macro_rules! gen_from_arrow_simple {
    ($($array:ident { $($native:ty:$arrow:ty)+ })+) => {
        $($(
            impl FromArrow for $native {
                type Array = $array<$arrow>;
                type Iter = ArrayIter<Self::Array, $native>;

                fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
                    Ok(ArrayIter::new(array))
                }
            }
        )+)+
    };
}

gen_from_arrow_simple!(
    PrimitiveArray {
        i8:Int8Type i16:Int16Type i32:Int32Type i64:Int64Type
        u8:UInt8Type u16:UInt16Type u32:UInt32Type u64:UInt64Type
        f16:Float16Type f32:Float32Type f64:Float64Type
    }

    GenericByteArray {
        Box<str>:Utf8Type String:Utf8Type
        Box<[u8]>:BinaryType Vec<u8>:BinaryType
        Large<Box<str>>:LargeUtf8Type Large<String>:LargeUtf8Type
        Large<Box<[u8]>>:LargeBinaryType Large<Vec<u8>>:LargeBinaryType
    }
);

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

// -----------------------------------------------------------------------------

pub struct Primitive<T: ArrowPrimitiveType>(pub T::Native);

impl<T: ArrowPrimitiveType> Wrapper<T::Native> for Primitive<T> {
    fn wrap(value: T::Native) -> Self {
        Primitive(value)
    }
}

impl<T: ArrowPrimitiveType> FromArrow for Primitive<T> {
    type Array = PrimitiveArray<T>;
    type Iter = Map<ArrayIter<Self::Array, T::Native>, FuncWrap<Primitive<T>>>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(map(ArrayIter::new(array)))
    }
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

impl<T, U> FromArrow for Bytes<T, U>
where
    T: ByteArrayType,
    for<'a> &'a GenericByteArray<T>: ArrayAccessor<Item=&'a T::Native>,
    for<'a> U: From<&'a T::Native>,
{
    type Array = GenericByteArray<T>;
    type Iter = ArrayIter<Self::Array, Bytes<T, U>>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(ArrayIter::new(array))
    }
}

// -----------------------------------------------------------------------------

pub struct FixedSizeBytes<T>(pub T);

impl<'a, T> From<&'a [u8]> for FixedSizeBytes<T>
where
    T: From<&'a [u8]>,
{
    fn from(value: &'a [u8]) -> FixedSizeBytes<T> {
        FixedSizeBytes(T::from(value))
    }
}

impl<T> FromArrow for FixedSizeBytes<T>
where
    for<'a> T: From<&'a [u8]>,
{
    type Array = FixedSizeBinaryArray;
    type Iter = ArrayIter<Self::Array, FixedSizeBytes<T>>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(ArrayIter::new(array))
    }
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

impl<T: FromArrow, O> Wrapper<Result<T::Iter>> for List<T, O> {
    fn wrap(value: Result<T::Iter>) -> Self {
        List::new(value)
    }
}

impl<T, O> FromArrow for List<T, O>
where
    T: FromArrow,
    O: OffsetSizeTrait,
{
    type Array = GenericListArray<O>;
    type Iter = Map<ArrayIter<Self::Array, ArrayRef>, Compose<FuncWrap<List<T, O>>, FuncFromAny<T>>>;

    fn from_nonnull_array(array: Self::Array) -> Result<Self::Iter> {
        Ok(map(ArrayIter::new(array)))
    }
}

// -----------------------------------------------------------------------------

pub struct Large<T>(pub T);

impl<'a, T> From<&'a str> for Large<T>
where
    T: From<&'a str>,
{
    fn from(value: &'a str) -> Self {
        Large(T::from(value))
    }
}

impl<'a, T> From<&'a [u8]> for Large<T>
where
    T: From<&'a [u8]>,
{
    fn from(value: &'a [u8]) -> Self {
        Large(T::from(value))
    }
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

pub trait Function<A> {
    type Output;
    fn call(args: A) -> Self::Output; 
}

pub struct Map<I, F>(I, PhantomData<F>);

impl<I, F> Iterator for Map<I, F>
where
    I: Iterator,
    F: Function<I::Item>,
{
    type Item = F::Output;

    fn next(&mut self) -> Option<F::Output> {
        Some(F::call(self.0.next()?))
    }
}

fn map<F, I>(iter: I) -> Map<I, F> {
    Map(iter, PhantomData)
}

// -----------------------------------------------------------------------------

pub struct FuncFromAny<T>(PhantomData<T>);

impl<T: FromArrow> Function<ArrayRef> for FuncFromAny<T> {
    type Output = Result<T::Iter>;

    fn call(args: ArrayRef) -> Self::Output {
        T::from_any(args)
    }
}

// -----------------------------------------------------------------------------

trait Wrapper<T> {
    fn wrap(value: T) -> Self;
}

pub struct FuncWrap<T>(PhantomData<T>);

impl<I, T: Wrapper<I>> Function<I> for FuncWrap<T> {
    type Output = T;

    fn call(args: I) -> Self::Output {
        T::wrap(args)
    }
}

// -----------------------------------------------------------------------------

pub struct Compose<F, G>(PhantomData<F>, PhantomData<G>);

impl<A, F, G> Function<A> for Compose<F, G>
where
    F: Function<G::Output>,
    G: Function<A>,
{
    type Output = F::Output;

    fn call(args: A) -> F::Output {
        F::call(G::call(args))
    }
}

// -----------------------------------------------------------------------------

fn null_check<T, A: Array>(array: &A) -> Result<()> {
    if has_nulls(array.nulls()) {
        Err(err_null::<T, A>())
    } else {
        Ok(())
    }
}

fn has_nulls(buf: Option<&NullBuffer>) -> bool {
    match buf {
        Some(nb) => nb.null_count() != 0,
        None => false,
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
