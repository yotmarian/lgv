use std::any::type_name;
use std::collections::VecDeque;

use arrow::array::*;
use arrow::buffer::NullBuffer;
use arrow::datatypes::*;

#[derive(Debug, Clone)]
pub enum Error {
    ContainsNulls,

    WrongArrowType {
        expected: &'static str,
    },

    WrongStructWidth {
        expected: usize,
        provided: usize,
    },
}

pub trait Array: Clone + Send + Sync + 'static {
    type Item<'a>;
    type Arrow: arrow::array::Array + Clone + 'static;

    fn len(&self) -> usize;
    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error>;
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item<'_>;

    fn from_arrow_ref(arrow: ArrayRef) -> Result<Self, Error> {
        match arrow.as_any().downcast_ref::<Self::Arrow>() {
            Some(arrow) => Self::from_arrow(arrow.clone()),
            None => Err(Error::WrongArrowType {
                expected: type_name::<Self::Arrow>()
            }),
        }
    }

    fn get(&self, i: usize) -> Option<Self::Item<'_>> {
        if i < self.len() {
            None
        } else {
            Some(unsafe { self.get_unchecked(i) })
        }
    }
}

impl<A: NullableArray> Array for A {
    type Item<'a> = Option<A::Item<'a>>;
    type Arrow = A::Arrow;

    fn len(&self) -> usize {
        self.nullable_len()
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        NullableArray::from_arrow(arrow)
    }

    fn from_arrow_ref(arrow: ArrayRef) -> Result<Self, Error> {
        NullableArray::from_arrow_ref(arrow)
    }

    unsafe fn get_unchecked(&self, i: usize) -> Option<A::Item<'_>> {
        nullable_get_unchecked(self, i)
    }
}

#[derive(Debug, Clone)]
pub struct NoNulls<A: NullableArray>(A);

impl<A: NullableArray> Array for NoNulls<A> {
    type Item<'a> = A::Item<'a>;
    type Arrow = A::Arrow;

    fn len(&self) -> usize {
        self.0.nullable_len()
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        null_check(&arrow)?;
        Ok(NoNulls(A::from_arrow(arrow)?))
    }

    fn from_arrow_ref(arrow: ArrayRef) -> Result<Self, Error> {
        null_check(&arrow)?;
        Ok(NoNulls(A::from_arrow_ref(arrow)?))
    }

    unsafe fn get_unchecked(&self, i: usize) -> Self::Item<'_> {
        self.0.get_unchecked_nonnull(i)
    }
}

// -----------------------------------------------------------------------------

pub trait NullableArray: Clone + Send + Sync + 'static {
    type Item<'a>;
    type Arrow: arrow::array::Array + Clone + 'static;

    fn nullable_len(&self) -> usize;
    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error>;
    unsafe fn is_null_unchecked(&self, i: usize) -> bool;
    unsafe fn get_unchecked_nonnull(&self, i: usize) -> Self::Item<'_>;

    fn from_arrow_dyn(arrow: &dyn arrow::array::Array) -> Result<Self, Error> {
        match arrow.as_any().downcast_ref::<Self::Arrow>() {
            Some(arrow) => Self::from_arrow(arrow.clone()),
            None => Err(Error::WrongArrowType {
                expected: type_name::<Self::Arrow>()
            }),
        }
    }

    fn from_arrow_ref(arrow: ArrayRef) -> Result<Self, Error> {
        Self::from_arrow_dyn(&arrow)
    }
}

impl NullableArray for ArrayRef {
    type Item<'a> = SomeItem<'a>;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    fn from_arrow_ref(arrow: ArrayRef) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> SomeItem {
        SomeItem { array: self, i }
    }
}

impl NullableArray for NullArray {
    type Item<'a> = Never;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, _: usize) -> bool {
        true
    }

    unsafe fn get_unchecked_nonnull(&self, _: usize) -> Never {
        panic!("Accesing element of NullArray")
    }
}

impl NullableArray for BooleanArray {
    type Item<'a> = bool;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> bool {
        self.value(i)
    }
}

impl<T: ArrowPrimitiveType> NullableArray for PrimitiveArray<T> {
    type Item<'a> = T::Native;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> T::Native {
        self.value(i)
    }
}

impl<T: ByteArrayType> NullableArray for GenericByteArray<T> {
    type Item<'a> = &'a T::Native;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> &T::Native {
        self.value(i)
    }
}

impl NullableArray for FixedSizeBinaryArray {
    type Item<'a> = &'a [u8];
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> &[u8] {
        self.value(i)
    }
}

impl<O: OffsetSizeTrait> NullableArray for GenericListArray<O> {
    type Item<'a> = ArrayRef;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> ArrayRef {
        self.value(i)
    }
}

impl NullableArray for FixedSizeListArray {
    type Item<'a> = ArrayRef;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> ArrayRef {
        self.value(i)
    }
}

impl NullableArray for StructArray {
    type Item<'a> = Box<[Option<SomeItem<'a>>]>;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> Box<[Option<SomeItem>]> {
        self.columns().iter()
            .map(|c| nullable_get_unchecked(c, i))
            .collect()
    }
}

impl NullableArray for UnionArray {
    type Item<'a> = SomeOwnedItem;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> SomeOwnedItem {
        SomeOwnedItem { array: self.value(i), i: 0 }
    }
}

impl NullableArray for MapArray {
    type Item<'a> = TypedStructArray<(ArrayRef, ArrayRef)>;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> Self::Item<'_> {
        NullableArray::from_arrow(self.value(i)).unwrap()
    }
}

impl<R: RunEndIndexType> NullableArray for RunArray<R> {
    type Item<'a> = SomeItem<'a>;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }
    
    unsafe fn get_unchecked_nonnull(&self, i: usize) -> SomeItem {
        SomeItem { array: self.values(), i: self.get_physical_index(i) }
    }
}

impl<K: ArrowDictionaryKeyType> NullableArray for DictionaryArray<K> {
    type Item<'a> = SomeItem<'a>;
    type Arrow = Self;

    fn nullable_len(&self) -> usize {
        arrow::array::Array::len(self)
    }

    fn from_arrow(arrow: Self::Arrow) -> Result<Self, Error> {
        Ok(arrow)
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(arrow::array::Array::nulls(self), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> SomeItem {
        SomeItem { array: self.values(), i: self.keys().value(i).as_usize() }
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TypedStructArray<C: Columns> {
    len: usize,
    nulls: Option<NullBuffer>,
    columns: C,
}

impl<C: Columns> NullableArray for TypedStructArray<C> {
    type Item<'a> = C::Items<'a>;
    type Arrow = StructArray;

    fn nullable_len(&self) -> usize {
        self.len
    }

    fn from_arrow(arrow: StructArray) -> Result<Self, Error> {
        let len = arrow.len();
        let (_, columns, nulls) = arrow.into_parts();

        Ok(TypedStructArray {
            len,
            nulls,
            columns: C::from_arrow(columns.into_boxed_slice())?
        })
    }

    unsafe fn is_null_unchecked(&self, i: usize) -> bool {
        is_null_unchecked(self.nulls.as_ref(), i)
    }

    unsafe fn get_unchecked_nonnull(&self, i: usize) -> C::Items<'_> {
        self.columns.get_unchecked(i)
    }
}

// -----------------------------------------------------------------------------

pub trait Columns: Clone + Send + Sync + 'static {
    type Items<'a>;

    fn from_arrow(columns: Box<[ArrayRef]>) -> Result<Self, Error>;
    unsafe fn get_unchecked(&self, i: usize) -> Self::Items<'_>;
}

impl<A: Array> Columns for A {
    type Items<'a> = A::Item<'a>;

    fn from_arrow(columns: Box<[ArrayRef]>) -> Result<Self, Error> {
        if columns.len() != 1 {
            Err(Error::WrongStructWidth {
                expected: 1,
                provided: columns.len(),
            })
        } else {
            let mut columns = Vec::from(columns);
            Ok(A::from_arrow_ref(columns.pop().unwrap())?)
        }
    }

    unsafe fn get_unchecked(&self, i: usize) -> A::Item<'_> {
        A::get_unchecked(self, i)
    }
}

impl Columns for () {
    type Items<'a> = ();

    fn from_arrow(_: Box<[ArrayRef]>) -> Result<Self, Error> {
        Ok(())
    }

    unsafe fn get_unchecked(&self, _: usize) -> () {
        ()
    }
}

macro_rules! impl_columns {
    { $($n:expr => ($($x:ident),+$(,)?))+ } => { $(
        #[allow(non_snake_case)]
        impl<$($x: Array),+> Columns for ($($x,)+) {
            type Items<'a> = ($($x::Item<'a>,)+);

            fn from_arrow(columns: Box<[ArrayRef]>) -> Result<Self, Error> {
                if columns.len() != $n {
                    Err(Error::WrongStructWidth {
                        expected: $n,
                        provided: columns.len(),
                    })
                } else {
                    let mut columns = VecDeque::from(Vec::from(columns));
                    Ok(($($x::from_arrow_ref(columns.pop_front().unwrap())?,)+))
                }
            }

            unsafe fn get_unchecked(&self, i: usize) -> Self::Items<'_> {
                let ($($x,)*) = self;
                ($($x.get_unchecked(i),)+)
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

// -----------------------------------------------------------------------------

pub enum Never { }

#[derive(Debug, Clone, Copy)]
pub struct SomeItem<'a> {
    array: &'a dyn arrow::array::Array,
    i: usize,
}

impl<'a> SomeItem<'a> {
    pub fn try_get<A: NullableArray>(self) -> Option<A::Item<'a>> {

        self.array.as_any().downcast_ref::<A>()
            .map(|a| unsafe { a.get_unchecked_nonnull(self.i) })
    }
}

#[derive(Debug, Clone)]
pub struct SomeOwnedItem {
    array: ArrayRef,
    i: usize,
}

impl SomeOwnedItem {
    pub fn item(&self) -> SomeItem {
        SomeItem { array: self.array.as_ref(), i: self.i }
    }
}

fn is_null_unchecked(nulls: Option<&NullBuffer>, i: usize) -> bool {
    nulls.map_or(false, |b| b.is_null(i))
}

unsafe fn nullable_get_unchecked<A: NullableArray>(array: &A, i: usize) -> Option<A::Item<'_>> {
    if array.is_null_unchecked(i) {
        None
    } else {
        Some(array.get_unchecked_nonnull(i))
    }
}

fn null_check<A: arrow::array::Array + ?Sized>(arrow: &A) -> Result<(), Error> {
    if arrow::array::Array::null_count(arrow) != 0 {
        Err(Error::ContainsNulls)
    } else {
        Ok(())
    }
}
