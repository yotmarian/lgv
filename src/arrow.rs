use std::any::type_name;
use std::fmt::{Debug, Display};
use std::path::PathBuf;

use arrow::array::{self as aa, ArrayRef};
use arrow::buffer::{NullBuffer, BooleanBuffer, ScalarBuffer};
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
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

macro_rules! impl_from_arrow {
    { $( ($($x:ident), +$(,)?) )+ } => { $(
        #[allow(non_snake_case)]
        impl<$($x: FromArrow),+> FromArrow for ($($x,)+) {
            type Array = StructArray<($($x::Array,)+)>;

            fn from_arrow(value: <Self::Array as Array>::Item<'_>) -> Self {
                let ($($x,)+) = value;
                ($($x::from_arrow($x),)+)
            }
        }
    )+ };
}

impl_from_arrow! {
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

// -----------------------------------------------------------------------------

pub trait DirectFromArrow: for<'a> From<<Self::Array as Array>::Item<'a>> {
    type Array: Array;
}

impl DirectFromArrow for () { type Array = UnitArray; }
impl DirectFromArrow for bool { type Array = BooleanArray; }
impl DirectFromArrow for Box<str> { type Array = StringArray; }
impl DirectFromArrow for String { type Array = StringArray; }
impl DirectFromArrow for PathBuf { type Array = StringArray; }
impl DirectFromArrow for Box<[u8]> { type Array = BinaryArray; }
impl DirectFromArrow for Vec<u8> { type Array = BinaryArray; }

impl<T: Primitive> DirectFromArrow for T {
    type Array = PrimitiveArray<T>;
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
            let x = T::from_arrow(unsafe { self.array.get_unchecked(self.i) });
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
    type Arrow;

    fn len(&self) -> usize;
    fn nullable_from_arrow(arrow: &Self::Arrow) -> Result<Nullable<Self>>;
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item<'_>;

    fn from_arrow(arrow: &Self::Arrow) -> Result<Self> {
        Self::nullable_from_arrow(arrow)?.null_check()
    }

    fn nullable_from_arrow_any(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        match arrow.as_any().downcast_ref::<Self::Arrow>() {
            Some(arrow) => Self::nullable_from_arrow(arrow),
            None => Err(Error::IncompatibleArrowType {
                expected: type_name::<Self::Arrow>(),
                provided: arrow.data_type().clone(),
            }),
        }
    }

    fn from_arrow_any(arrow: &dyn aa::Array) -> Result<Self> {
        Self::nullable_from_arrow_any(arrow)?.null_check()
    }

    fn nullable_from_arrow_batch(arrow: &RecordBatch) -> Result<Nullable<Self>> {
        let [column] = n_columns(arrow.columns())?;
        Self::nullable_from_arrow_any(column)
    }

    fn from_arrow_batch(arrow: &RecordBatch) -> Result<Self> {
        Self::nullable_from_arrow_batch(arrow)?.null_check()
    }

    fn get(&self, i: usize) -> Option<Self::Item<'_>> {
        if i < self.len() {
            None
        } else {
            Some(unsafe { self.get_unchecked(i) })
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
    type Arrow = A::Arrow;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn nullable_from_arrow(arrow: &Self::Arrow) -> Result<Nullable<Self>> {
        Ok(Nullable(A::nullable_from_arrow(arrow)?, None))
    }

    fn from_arrow(arrow: &Self::Arrow) -> Result<Self> {
        A::nullable_from_arrow(arrow)
    }

    unsafe fn get_unchecked(&self, i: usize) -> Self::Item<'_> {
        match &self.1 {
            Some(b) if b.is_null(i) => None,
            _ => Some(self.0.get_unchecked(i)),
        }
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct UnitArray(usize);

impl Array for UnitArray {
    type Item<'a> = ();
    type Arrow = !;

    fn len(&self) -> usize {
        self.0
    }

    fn nullable_from_arrow(arrow: &!) -> Result<Nullable<Self>> {
        match *arrow { }
    }

    fn nullable_from_arrow_any(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        Ok(Nullable(Self(arrow.len()), None))
    }

    fn nullable_from_arrow_batch(arrow: &RecordBatch) -> Result<Nullable<Self>> {
        Ok(Nullable(Self(arrow.num_rows()), None))
    }

    unsafe fn get_unchecked(&self, _: usize) -> () {
        ()
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BooleanArray(BooleanBuffer);

impl Array for BooleanArray {
    type Item<'a> = bool;
    type Arrow = aa::BooleanArray;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn nullable_from_arrow(arrow: &Self::Arrow) -> Result<Nullable<Self>> {
        let nulls = clone_nulls(arrow);
        Ok(Nullable(Self(arrow.values().clone()), nulls))
    }

    unsafe fn get_unchecked(&self, i: usize) -> bool {
        self.0.value(i)
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug)]
pub struct PrimitiveArray<T: Primitive>(
    ScalarBuffer<<T::Arrow as ArrowPrimitiveType>::Native>
);

impl<T: Primitive> Clone for PrimitiveArray<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Primitive> Array for PrimitiveArray<T> {
    type Item<'a> = T;
    type Arrow = aa::PrimitiveArray<T::Arrow>;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn nullable_from_arrow(arrow: &Self::Arrow) -> Result<Nullable<Self>> {
        Ok(Nullable(Self(arrow.values().clone()), clone_nulls(arrow)))
    }

    unsafe fn get_unchecked(&self, i: usize) -> T {
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
    type Arrow = aa::GenericByteArray<T>;

    fn len(&self) -> usize {
        aa::Array::len(&self.0)
    }

    fn nullable_from_arrow(arrow: &Self::Arrow) -> Result<Nullable<Self>> {
        let nulls = clone_nulls(arrow);
        Ok(Nullable(Self(arrow.clone()), nulls))
    }

    unsafe fn get_unchecked(&self, i: usize) -> &T::Native {
        self.0.value(i)
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SomeFixedSizeBinaryArray(aa::FixedSizeBinaryArray);

impl Array for SomeFixedSizeBinaryArray {
    type Item<'a> = &'a [u8];
    type Arrow = aa::FixedSizeBinaryArray;

    fn len(&self) -> usize {
        aa::Array::len(&self.0)
    }

    fn nullable_from_arrow(arrow: &Self::Arrow) -> Result<Nullable<Self>> {
        let nulls = clone_nulls(arrow);
        Ok(Nullable(Self(arrow.clone()), nulls))
    }

    unsafe fn get_unchecked(&self, i: usize) -> &[u8] {
        self.0.value(i)
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FixedSizeBinaryArray<const N: usize>(SomeFixedSizeBinaryArray);

impl<const N: usize> Array for FixedSizeBinaryArray<N> {
    type Item<'a> = &'a [u8; N];
    type Arrow = aa::FixedSizeBinaryArray;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn nullable_from_arrow(arrow: &Self::Arrow) -> Result<Nullable<Self>> {
        if arrow.value_length().try_into() != Ok(N) {
            return Err(Error::IncompatibleFixedSize {
                expected: N,
                provided: arrow.value_length(),
            });
        }

        Ok(SomeFixedSizeBinaryArray::nullable_from_arrow(arrow)?.map(Self))
    }

    unsafe fn get_unchecked(&self, i: usize) -> &[u8; N] {
        self.0.get_unchecked(i).try_into().unwrap()
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum StringArray {
    Normal(GenericByteArray<Utf8Type>),
    Large(GenericByteArray<LargeUtf8Type>),
}

impl Array for StringArray {
    type Item<'a> = &'a str;
    type Arrow = !;

    fn len(&self) -> usize {
        match self {
            Self::Normal(n) => n.len(),
            Self::Large(l) => l.len(),
        }
    }

    fn nullable_from_arrow(arrow: &!) -> Result<Nullable<Self>> {
        match *arrow { }
    }

    fn nullable_from_arrow_any(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        if let Ok(n) = GenericByteArray::<Utf8Type>::nullable_from_arrow_any(arrow) {
            return Ok(n.map(Self::Normal))
        }

        if let Ok(n) = GenericByteArray::<LargeUtf8Type>::nullable_from_arrow_any(arrow) {
            return Ok(n.map(Self::Large))
        }

        Err(Error::IncompatibleArrowType {
            expected: "ByteArray of Utf8Type or LargeUtf8Type",
            provided: arrow.data_type().clone(),
        })
    }

    unsafe fn get_unchecked(&self, i: usize) -> &str {
        match self {
            Self::Normal(n) => n.get_unchecked(i),
            Self::Large(l) => l.get_unchecked(i),
        }
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum BinaryArray {
    Normal(GenericByteArray<BinaryType>),
    Large(GenericByteArray<LargeBinaryType>),
    FixedSize(SomeFixedSizeBinaryArray),
}

impl Array for BinaryArray {
    type Item<'a> = &'a [u8];
    type Arrow = !;

    fn len(&self) -> usize {
        match self {
            Self::Normal(n) => n.len(),
            Self::Large(l) => l.len(),
            Self::FixedSize(f) => f.len(),
        }
    }

    fn nullable_from_arrow(arrow: &!) -> Result<Nullable<Self>> {
        match *arrow { }
    }

    fn nullable_from_arrow_any(arrow: &dyn aa::Array) -> Result<Nullable<Self>> {
        if let Ok(n) = GenericByteArray::<BinaryType>::nullable_from_arrow_any(arrow) {
            return Ok(n.map(Self::Normal))
        }

        if let Ok(n) = GenericByteArray::<LargeBinaryType>::nullable_from_arrow_any(arrow) {
            return Ok(n.map(Self::Large))
        }

        if let Ok(n) = SomeFixedSizeBinaryArray::nullable_from_arrow_any(arrow) {
            return Ok(n.map(Self::FixedSize))
        }

        Err(Error::IncompatibleArrowType {
            expected: "FixedSizeBinaryArray or ByteArray of BinaryType or LargeBinaryType",
            provided: arrow.data_type().clone(),
        })
    }

    unsafe fn get_unchecked(&self, i: usize) -> &[u8] {
        match self {
            Self::Normal(n) => n.get_unchecked(i),
            Self::Large(l) => l.get_unchecked(i),
            Self::FixedSize(f) => f.get_unchecked(i),
        }
    }
}

// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StructArray<C: StructColumns> {
    len: usize,
    columns: C,
}

impl<C: StructColumns> Array for StructArray<C> {
    type Item<'a> = C::Items<'a>;
    type Arrow = aa::StructArray;

    fn len(&self) -> usize {
        self.len
    }

    fn nullable_from_arrow(arrow: &Self::Arrow) -> Result<Nullable<Self>> {
        Ok(Nullable(
            StructArray {
                len: aa::Array::len(arrow),
                columns: C::columns_from_arrow(arrow.columns())?
            },
            clone_nulls(arrow),
        ))
    }

    fn nullable_from_arrow_batch(arrow: &RecordBatch) -> Result<Nullable<Self>> {
        Ok(Nullable(Self::from_arrow_batch(arrow)?, None))
    }

    fn from_arrow_batch(arrow: &RecordBatch) -> Result<Self> {
        Ok(StructArray {
            len: arrow.num_rows(),
            columns: C::columns_from_arrow(arrow.columns())?,
        })
    }

    unsafe fn get_unchecked(&self, i: usize) -> C::Items<'_> {
        self.columns.columns_get_unchecked(i)
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
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl std::error::Error for Error { }

// -----------------------------------------------------------------------------

#[derive(Debug)]
pub struct ConversionError {
    pub from: &'static str,
    pub to: &'static str,
    pub source: Option<Box<dyn std::error::Error + 'static>>,
}

impl Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl std::error::Error for ConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref()
    }
}

// -----------------------------------------------------------------------------

pub trait Primitive: From<<Self::Arrow as ArrowPrimitiveType>::Native> + 'static {
    type Arrow: ArrowPrimitiveType;
}

macro_rules! impl_primitive {
    { $($native:ty:$arrow:ty)+ } => {
        $(impl Primitive for $native { type Arrow = $arrow; })+
    };
}

impl_primitive! {
    i8:Int8Type i16:Int16Type i32:Int32Type i64:Int64Type
    u8:UInt8Type u16:UInt16Type u32:UInt32Type u64:UInt64Type
    f16:Float16Type f32:Float32Type f64:Float64Type
}

// -----------------------------------------------------------------------------

pub trait StructColumns: Clone + Send + Sync + 'static {
    type Items<'a>;

    fn columns_from_arrow(columns: &[ArrayRef]) -> Result<Self>;
    unsafe fn columns_get_unchecked(&self, i: usize) -> Self::Items<'_>;
}

impl StructColumns for () {
    type Items<'a> = ();

    fn columns_from_arrow(_: &[ArrayRef]) -> Result<Self> {
        Ok(())
    }

    unsafe fn columns_get_unchecked(&self, _: usize) -> () {
        ()
    }
}

impl<A: Array> StructColumns for A {
    type Items<'a> = A::Item<'a>;

    fn columns_from_arrow(columns: &[ArrayRef]) -> Result<Self> {
        let [column] = n_columns(columns)?;
        Ok(A::from_arrow_any(column)?)
    }

    unsafe fn columns_get_unchecked(&self, i: usize) -> A::Item<'_> {
        A::get_unchecked(self, i)
    }
}

macro_rules! impl_struct_columns {
    { $(($($x:ident),+$(,)?))+ } => { $(
        #[allow(non_snake_case)]
        impl<$($x: Array),+> StructColumns for ($($x,)+) {
            type Items<'a> = ($($x::Item<'a>,)+);

            fn columns_from_arrow(columns: &[ArrayRef]) -> Result<Self> {
                let [$($x),+] = n_columns(columns)?;
                Ok(($($x::from_arrow_any($x)?,)+))
            }

            unsafe fn columns_get_unchecked(&self, i: usize) -> Self::Items<'_> {
                let ($($x,)+) = self;
                ($($x.get_unchecked(i),)+)
            }
        }
    )+ };
}

impl_struct_columns! {
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

// -----------------------------------------------------------------------------

fn clone_nulls<A: aa::Array>(array: &A) -> Option<NullBuffer> {
    array.nulls().map(|b| b.clone())
}

fn union_nulls(x: Option<NullBuffer>, y: Option<NullBuffer>) -> Option<NullBuffer> {
    match (x, y) {
        (Some(x), Some(y)) => Some(NullBuffer::new(x.inner() & y.inner())),
        (x, None) => x,
        (None, y) => y,
    }
}

fn n_columns<const N: usize>(columns: &[ArrayRef]) -> Result<&[ArrayRef; N]> {
    columns.try_into()
        .map_err(|_| Error::IncompatibleStructWidth {
            expected: 1,
            provided: columns.len(),
        })
}
