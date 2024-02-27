use std::{
    any::type_name,
    collections::HashSet,
    fmt::{Debug, Display},
    ops::Range,
    path::PathBuf,
};

use arrow::{
    array::{self as aa, ArrayRef, OffsetSizeTrait},
    buffer::{BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::*,
    record_batch::RecordBatch,
};
use half::f16;

pub trait FromArrow {
    type Array: Array;

    fn from_arrow(value: <Self::Array as Array>::Item<'_>) -> Self;
}

impl<T: DirectFromArrow> FromArrow for T {
    type Array = T::Array;

    fn from_arrow(value: <Self::Array as Array>::Item<'_>) -> Self {
        T::from(value)
    }
}

impl<T: FromArrow> FromArrow for Option<T> {
    type Array = Nullable<T::Array>;

    fn from_arrow(value: <Self::Array as Array>::Item<'_>) -> Self {
        value.map(T::from_arrow)
    }
}

macro_rules! impl_from_arrow_tuple {
    { $( ($($x:ident), +$(,)?): $struct:ident )+ } => { $(
        #[allow(non_snake_case, unused_parens)]
        impl<$($x: FromArrow),+> FromArrow for ($($x,)+) {
            type Array = $struct<$($x::Array),+>;

            fn from_arrow(value: <Self::Array as Array>::Item<'_>) -> Self {
                let ($($x),+) = value;
                ($($x::from_arrow($x),)+)
            }
        }
    )+ };
}

impl_from_arrow_tuple! {
    (A,): Struct1Array
    (A, B): Struct2Array
    (A, B, C): Struct3Array
    (A, B, C, D): Struct4Array
    (A, B, C, D, E): Struct5Array
    (A, B, C, D, E, F): Struct6Array
    (A, B, C, D, E, F, G): Struct7Array
    (A, B, C, D, E, F, G, H): Struct8Array
    (A, B, C, D, E, F, G, H, I): Struct9Array
    (A, B, C, D, E, F, G, H, I, J): Struct10Array
    (A, B, C, D, E, F, G, H, I, J, K): Struct11Array
    (A, B, C, D, E, F, G, H, I, J, K, L): Struct12Array
}

// -----------------------------------------------------------------------------

pub trait DirectFromArrow:
    for<'a> From<<Self::Array as Array>::Item<'a>>
{
    type Array: Array;
}

macro_rules! impl_direct_from_arrow {
    { $($native:ty: $arrow:ty)+ } => { $(
        impl DirectFromArrow for $native { type Array = $arrow; }
    )+ };
}

impl_direct_from_arrow! {
    (): UnitArray
    bool: BooleanArray
    Box<str>: StringArray
    String: StringArray
    PathBuf: StringArray
    Box<[u8]>: BinaryArray
    Vec<u8>: BinaryArray
}

impl<T: Primitive> DirectFromArrow for T {
    type Array = PrimitiveArray<T>;
}

// -----------------------------------------------------------------------------

pub trait Primitive:
    From<<Self::Arrow as ArrowPrimitiveType>::Native> + 'static
{
    type Arrow: ArrowPrimitiveType;
}

macro_rules! impl_primitive {
    { $( $native:ty: $arrow:ty )+ } => { $(
        impl Primitive for $native { type Arrow = $arrow; }
    )+ };
}

impl_primitive! {
    i8:Int8Type i16:Int16Type i32:Int32Type i64:Int64Type
    u8:UInt8Type u16:UInt16Type u32:UInt32Type u64:UInt64Type
    f16:Float16Type f32:Float32Type f64:Float64Type
}

// -----------------------------------------------------------------------------

pub struct Iter<T: FromArrow> {
    array: T::Array,
    i: usize,
}

impl<T: FromArrow> Iter<T> {
    pub fn new(array: T::Array) -> Iter<T> {
        Iter { array, i: 0 }
    }
}

impl<T: FromArrow> Iterator for Iter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.array.len() {
            let x = T::from_arrow(self.array.index(self.i));
            self.i += 1;
            Some(x)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<T: FromArrow> ExactSizeIterator for Iter<T> {
    fn len(&self) -> usize {
        self.array.len() - self.i
    }
}

// -----------------------------------------------------------------------------

pub trait Array: Clone + Send + Sync + 'static {
    type Item<'a>;

    fn len(&self) -> usize;
    fn slice(&self, offset: usize, len: usize) -> Self;
    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>>;
    fn index(&self, i: usize) -> Self::Item<'_>;

    fn from_arrow(arrow: &dyn aa::Array) -> Result<Self> {
        Self::nullable_from_arrow(arrow)?.null_check()
    }

    fn nullable_from_arrow_batch(
        arrow: &RecordBatch,
    ) -> Result<Nullable<Self>> {
        let [column] = n_columns(arrow.columns())?;
        Self::nullable_from_arrow(column)
    }

    fn from_arrow_batch(arrow: &RecordBatch) -> Result<Self> {
        Self::nullable_from_arrow_batch(arrow)?.null_check()
    }

    fn get(&self, i: usize) -> Option<Self::Item<'_>> {
        if i < self.len() {
            None
        } else {
            Some(self.index(i))
        }
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Nullable<A>(A, Option<NullBuffer>);

impl<A> Nullable<A> {
    pub fn map<B, F: FnMut(A) -> B>(self, mut f: F) -> Nullable<B> {
        Nullable(f(self.0), self.1)
    }

    pub fn no_nulls(self) -> Option<A> {
        match self.1 {
            Some(b) if b.null_count() != 0 => None,
            _ => Some(self.0),
        }
    }

    pub fn union_nulls(self, other: Option<NullBuffer>) -> Self {
        Self(self.0, union_nulls(self.1, other))
    }

    fn null_check(self) -> Result<A> {
        self.no_nulls().ok_or(Error::ContainsNulls)
    }
}

impl<A: Array> Array for Nullable<A> {
    type Item<'a> = Option<A::Item<'a>>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self(
            self.0.slice(offset, len),
            self.1.as_ref().map(|b| b.slice(offset, len)),
        )
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        Ok(Nullable(A::nullable_from_arrow(arrow)?, None))
    }

    fn from_arrow(arrow: &dyn aa::Array) -> Result<Self> {
        A::nullable_from_arrow(arrow)
    }

    fn index(&self, i: usize) -> Self::Item<'_> {
        match &self.1 {
            Some(b) if b.is_null(i) => None,
            _ => Some(self.0.index(i)),
        }
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UnitArray(usize);

impl Array for UnitArray {
    type Item<'a> = ();

    fn len(&self) -> usize {
        self.0
    }

    fn slice(&self, _: usize, len: usize) -> Self {
        Self(len)
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        Ok(Nullable(Self(arrow.len()), None))
    }

    fn nullable_from_arrow_batch(
        arrow: &RecordBatch,
    ) -> Result<Nullable<Self>> {
        Ok(Nullable(Self(arrow.num_rows()), None))
    }

    fn index(&self, _: usize) -> () {
        ()
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BooleanArray(BooleanBuffer);

impl Array for BooleanArray {
    type Item<'a> = bool;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self(self.0.slice(offset, len))
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        let arrow: &aa::BooleanArray = downcast(arrow)?;
        let nulls = clone_nulls(arrow);
        Ok(Nullable(Self(arrow.values().clone()), nulls))
    }

    fn index(&self, i: usize) -> bool {
        self.0.value(i)
    }
}

// -----------------------------------------------------------------------------

pub struct PrimitiveArray<T: Primitive>(
    ScalarBuffer<<T::Arrow as ArrowPrimitiveType>::Native>,
);

impl<T: Primitive> Debug for PrimitiveArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PrimitiveArray").field(&self.0).finish()
    }
}

impl<T: Primitive> Clone for PrimitiveArray<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Primitive> Array for PrimitiveArray<T> {
    type Item<'a> = T;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self(self.0.slice(offset, len))
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        let arrow: &aa::PrimitiveArray<T::Arrow> = downcast(arrow)?;
        Ok(Nullable(Self(arrow.values().clone()), clone_nulls(arrow)))
    }

    fn index(&self, i: usize) -> T {
        T::from(self.0[i])
    }
}

// -----------------------------------------------------------------------------

pub struct GenericByteArray<T: ByteArrayType>(aa::GenericByteArray<T>);

impl<T: ByteArrayType> Debug for GenericByteArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("GenericByteArray").field(&self.0).finish()
    }
}

impl<T: ByteArrayType> Clone for GenericByteArray<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: ByteArrayType> Array for GenericByteArray<T> {
    type Item<'a> = &'a T::Native;

    fn len(&self) -> usize {
        aa::Array::len(&self.0)
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self(self.0.slice(offset, len))
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        let arrow: &aa::GenericByteArray<T> = downcast(arrow)?;
        let nulls = clone_nulls(arrow);
        Ok(Nullable(Self(arrow.clone()), nulls))
    }

    fn index(&self, i: usize) -> &T::Native {
        self.0.value(i)
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SomeFixedSizeBinaryArray(aa::FixedSizeBinaryArray);

impl Array for SomeFixedSizeBinaryArray {
    type Item<'a> = &'a [u8];

    fn len(&self) -> usize {
        aa::Array::len(&self.0)
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self(self.0.slice(offset, len))
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        let arrow: &aa::FixedSizeBinaryArray = downcast(arrow)?;
        let nulls = clone_nulls(arrow);
        Ok(Nullable(Self(arrow.clone()), nulls))
    }

    fn index(&self, i: usize) -> &[u8] {
        self.0.value(i)
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FixedSizeBinaryArray<const N: usize>(SomeFixedSizeBinaryArray);

impl<const N: usize> Array for FixedSizeBinaryArray<N> {
    type Item<'a> = &'a [u8; N];

    fn len(&self) -> usize {
        self.0.len()
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self(self.0.slice(offset, len))
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        let some = SomeFixedSizeBinaryArray::nullable_from_arrow(arrow)?;

        if some.0 .0.value_length().try_into() != Ok(N) {
            return Err(Error::IncompatibleFixedSize {
                expected: N,
                provided: some.0 .0.value_length(),
            });
        }

        Ok(some.map(Self))
    }

    fn index(&self, i: usize) -> &[u8; N] {
        self.0.index(i).try_into().unwrap()
    }
}

// -----------------------------------------------------------------------------

pub struct GenericListArray<O: OffsetSizeTrait, A: Array> {
    inner: A,
    offsets: OffsetBuffer<O>,
}

impl<O: OffsetSizeTrait, A: Array> GenericListArray<O, A> {
    fn item_range(&self, i: usize) -> Range<usize> {
        let end = self.offsets[i + 1].as_usize();
        let start = self.offsets[i].as_usize();
        start .. end
    }

    fn sizes(&self) -> impl Iterator<Item = usize> + '_ {
        self.offsets
            .iter()
            .map_windows(|&[c, n]| n.as_usize() - c.as_usize())
    }

    fn check_fixed_size(&self, size: usize) -> Result<()> {
        if self.sizes().all(|s| s == size) {
            Ok(())
        } else {
            Err(Error::SizeNotFixed {
                expected: size,
                provided: self.sizes().collect(),
            })
        }
    }
}

impl<O: OffsetSizeTrait, A: Array + Debug> Debug for GenericListArray<O, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenericListArray")
            .field("inner", &self.inner)
            .field("offsets", &self.offsets)
            .finish()
    }
}

impl<O: OffsetSizeTrait, A: Array> Clone for GenericListArray<O, A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            offsets: self.offsets.clone(),
        }
    }
}

impl<O: OffsetSizeTrait, A: Array> Array for GenericListArray<O, A> {
    type Item<'a> = A;

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self {
            inner: self.inner.clone(),
            offsets: self.offsets.slice(offset, len),
        }
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        let arrow: &aa::GenericListArray<O> = downcast(arrow)?;
        let inner = A::from_arrow(arrow.values())?;
        let offsets = arrow.offsets().clone();
        let nulls = clone_nulls(arrow);
        Ok(Nullable(Self { inner, offsets }, nulls))
    }

    fn index(&self, i: usize) -> Self::Item<'_> {
        let loc = self.item_range(i);
        self.inner.slice(loc.start, loc.end - loc.start)
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FixedSizeListArray<A> {
    len: usize,
    size: i32,
    inner: A,
}

impl<A> FixedSizeListArray<A> {
    fn item_range(&self, i: usize) -> Range<usize> {
        let start = i * self.size as usize;
        start .. start + self.size as usize
    }

    fn check_fixed_size(&self, size: usize) -> Result<()> {
        if self.size.try_into() == Ok(size) {
            Ok(())
        } else {
            Err(Error::IncompatibleFixedSize {
                expected: size,
                provided: self.size,
            })
        }
    }
}

impl<A: Array> Array for FixedSizeListArray<A> {
    type Item<'a> = A;

    fn len(&self) -> usize {
        self.len
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        let s = self.size as usize;

        Self {
            len,
            size: self.size,
            inner: self.inner.slice(offset * s, len * s),
        }
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        let arrow: &aa::FixedSizeListArray = downcast(arrow)?;
        let len = aa::Array::len(arrow);
        let size = arrow.value_length();
        let inner = A::from_arrow(arrow.values())?;
        let nulls = clone_nulls(arrow);
        Ok(Nullable(Self { len, size, inner }, nulls))
    }

    fn index(&self, i: usize) -> Self::Item<'_> {
        self.inner.slice(i * self.size as usize, self.size as usize)
    }
}

// -----------------------------------------------------------------------------

macro_rules! gen_struct_arrays {
    { $( $name:ident<$($x:ident),+> )+ } => { $(
        #[derive(Debug, Clone)]
        pub struct $name<$($x),+> {
            len: usize,
            columns: ($($x,)+),
        }

        #[allow(non_snake_case)]
        impl<$($x: Array),+> $name<$($x),+> {
            fn new(len: usize, columns: &[ArrayRef]) -> Result<Self> {
                let [$($x),+] = n_columns(columns)?;
                Ok(Self {
                    len,
                    columns: ($($x::from_arrow($x)?,)+),
                })
            }
        }

        #[allow(unused_parens, non_snake_case)]
        impl<$($x: Array),+> Array for $name<$($x),+> {
            type Item<'a> = ($($x::Item<'a>),+);

            fn len(&self) -> usize {
                self.len
            }

            fn slice(&self, offset: usize, len: usize) -> Self {
                let ($($x,)+) = &self.columns;

                Self {
                    len,
                    columns: ($($x.slice(offset, len),)+),
                }
            }

            fn nullable_from_arrow(
                arrow: &dyn aa::Array,
            ) -> Result<Nullable<Self>> {
                let arrow: &aa::StructArray = downcast(arrow)?;
                let nulls = clone_nulls(arrow);
                Ok(Nullable(
                    Self::new(aa::Array::len(arrow), arrow.columns())?,
                    nulls,
                ))
            }

            fn from_arrow_batch(arrow: &RecordBatch) -> Result<Self> {
                Self::new(arrow.num_rows(), arrow.columns())
            }

            fn nullable_from_arrow_batch(
                arrow: &RecordBatch,
            ) -> Result<Nullable<Self>> {
                Ok(Nullable(Self::from_arrow_batch(arrow)?, None))
            }

            fn index(&self, i: usize) -> Self::Item<'_> {
                let ($($x,)+) = &self.columns;
                ($($x.index(i)),+)
            }
        }
    )+ };
}

gen_struct_arrays! {
    Struct1Array<A>
    Struct2Array<A, B>
    Struct3Array<A, B, C>
    Struct4Array<A, B, C, D>
    Struct5Array<A, B, C, D, E>
    Struct6Array<A, B, C, D, E, F>
    Struct7Array<A, B, C, D, E, F, G>
    Struct8Array<A, B, C, D, E, F, G, H>
    Struct9Array<A, B, C, D, E, F, G, H, I>
    Struct10Array<A, B, C, D, E, F, G, H, I, J>
    Struct11Array<A, B, C, D, E, F, G, H, I, J, K>
    Struct12Array<A, B, C, D, E, F, G, H, I, J, K, L>
}

// -----------------------------------------------------------------------------

macro_rules! gen_one_ofs {
    { $( $name:ident<$x:ident, $($xs:ident),+> )+ } => { $(
        #[derive(Debug, Clone)]
        pub enum $name<$x, $($xs),+> {
            $x($x),
            $($xs($xs)),+
        }

        #[allow(non_snake_case)]
        impl<$x: Array, $($xs),+> Array for $name<$x, $($xs),+>
        where
            $($xs: for<'a> Array<Item<'a>=$x::Item<'a>>),+
        {
            type Item<'a> = $x::Item<'a>;

            fn len(&self) -> usize {
                match self {
                    Self::$x(x) => x.len(),
                    $(Self::$xs(x) => x.len()),+
                }
            }

            fn slice(&self, offset: usize, len: usize) -> Self {
                match self {
                    Self::$x(x) => Self::$x(x.slice(offset, len)),
                    $(Self::$xs(x) => Self::$xs(x.slice(offset, len))),+
                }
            }

            fn nullable_from_arrow(
                arrow: &dyn aa::Array,
            ) -> Result<Nullable<Self>> {
                let $x = match $x::nullable_from_arrow(arrow) {
                    Ok(x) => return Ok(x.map(Self::$x)),
                    Err(e) => e,
                };

                $(
                    let $xs = match $xs::nullable_from_arrow(arrow) {
                        Ok(x) => return Ok(x.map(Self::$xs)),
                        Err(e) => e,
                    };
                )+

                Err(Error::Multiple {
                    context: type_name::<Self>(),
                    errors: vec![$x, $($xs),+],
                })
            }

            fn index(&self, i: usize) -> Self::Item<'_> {
                match self {
                    Self::$x(x) => x.index(i),
                    $(Self::$xs(x) => x.index(i)),+
                }
            }
        }
    )+ };
}

gen_one_ofs! {
    OneOf2<A, B>
    OneOf3<A, B, C>
}

// -----------------------------------------------------------------------------

pub type StringArray =
    OneOf2<GenericByteArray<Utf8Type>, GenericByteArray<LargeUtf8Type>>;

pub type BinaryArray = OneOf3<
    GenericByteArray<BinaryType>,
    GenericByteArray<LargeBinaryType>,
    SomeFixedSizeBinaryArray,
>;

pub type ListArray<A> = OneOf3<
    GenericListArray<i32, A>,
    GenericListArray<i64, A>,
    FixedSizeListArray<A>,
>;

// -----------------------------------------------------------------------------

pub struct PrimitiveListArray<T: Primitive>(ListArray<PrimitiveArray<T>>);

impl<T: Primitive> Debug for PrimitiveListArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PrimitiveListArray").field(&self.0).finish()
    }
}

impl<T: Primitive> Clone for PrimitiveListArray<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Array for PrimitiveListArray<T>
where
    T: Primitive,
    T: ArrowNativeType,
    T::Arrow: ArrowPrimitiveType<Native = T>,
{
    type Item<'a> = &'a [T];

    fn len(&self) -> usize {
        self.0.len()
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self(self.0.slice(offset, len))
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        Ok(Array::nullable_from_arrow(arrow)?.map(Self))
    }

    fn index(&self, i: usize) -> Self::Item<'_> {
        let (prim, r) = match &self.0 {
            OneOf3::A(x) => (&x.inner, x.item_range(i)),
            OneOf3::B(x) => (&x.inner, x.item_range(i)),
            OneOf3::C(x) => (&x.inner, x.item_range(i)),
        };

        &prim.0[r]
    }
}

// -----------------------------------------------------------------------------

pub struct LenientFixedSizePrimitiveListArray<T: Primitive, const N: usize>(
    PrimitiveListArray<T>,
);

impl<T: Primitive, const N: usize> Debug
    for LenientFixedSizePrimitiveListArray<T, N>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("LenientFixedSizePrimitiveListArray")
            .field(&self.0)
            .finish()
    }
}

impl<T: Primitive, const N: usize> Clone
    for LenientFixedSizePrimitiveListArray<T, N>
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T, const N: usize> Array for LenientFixedSizePrimitiveListArray<T, N>
where
    T: Primitive,
    T: ArrowNativeType,
    T::Arrow: ArrowPrimitiveType<Native = T>,
{
    type Item<'a> = &'a [T; N];

    fn len(&self) -> usize {
        self.0.len()
    }

    fn slice(&self, offset: usize, len: usize) -> Self {
        Self(self.0.slice(offset, len))
    }

    fn nullable_from_arrow(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        let inner = PrimitiveListArray::<T>::nullable_from_arrow(arrow)?;

        match &inner.0 .0 {
            OneOf3::A(x) => x.check_fixed_size(N),
            OneOf3::B(x) => x.check_fixed_size(N),
            OneOf3::C(x) => x.check_fixed_size(N),
        }?;

        Ok(inner.map(Self))
    }

    fn index(&self, i: usize) -> Self::Item<'_> {
        self.0.index(i).try_into().unwrap()
    }
}

// -----------------------------------------------------------------------------

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    ContainsNulls,

    IncompatibleFixedSize {
        expected: usize,
        provided: i32,
    },

    IncompatibleArrowType {
        expected: &'static str,
        provided: DataType,
    },

    IncompatibleStructWidth {
        expected: usize,
        provided: usize,
    },

    SizeNotFixed {
        expected: usize,
        provided: HashSet<usize>,
    },

    Multiple {
        context: &'static str,
        errors: Vec<Error>,
    },
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl std::error::Error for Error {}

// -----------------------------------------------------------------------------

fn clone_nulls<A: aa::Array>(array: &A) -> Option<NullBuffer> {
    array.nulls().map(|b| b.clone())
}

fn union_nulls(
    x: Option<NullBuffer>,
    y: Option<NullBuffer>,
) -> Option<NullBuffer> {
    match (x, y) {
        (Some(x), Some(y)) => Some(NullBuffer::new(x.inner() & y.inner())),
        (x, None) => x,
        (None, y) => y,
    }
}

fn downcast<A: 'static>(arrow: &dyn aa::Array) -> Result<&A> {
    match arrow.as_any().downcast_ref::<A>() {
        Some(arrow) => Ok(arrow),
        None => Err(Error::IncompatibleArrowType {
            expected: type_name::<A>(),
            provided: arrow.data_type().clone(),
        }),
    }
}

fn n_columns<const N: usize>(columns: &[ArrayRef]) -> Result<&[ArrayRef; N]> {
    columns
        .try_into()
        .map_err(|_| Error::IncompatibleStructWidth {
            expected: 1,
            provided: columns.len(),
        })
}
