//! Named slicing, xarray-style: dims are addressed by name, in any order,
//! and unmentioned dims are kept whole. Specs are built with the [`s!`]
//! macro and applied with `slice`/`slice_assign`/`slice_fill` on both the
//! typed and untyped [`NamedTensor`]s.

use crate::typed::DimName;
use std::marker::PhantomData;

pub use burn::tensor::Slice;

/// A slice bound to a compile-time dim name; which axis it applies to is
/// decided only when the spec meets a tensor.
#[derive(Clone, Debug)]
pub struct DimSlice<D> {
    pub(crate) slice: Slice,
    _dim: PhantomData<D>,
}

impl<D: DimName> DimSlice<D> {
    pub fn new<S: Into<Slice>>(_dim: D, slice: S) -> Self {
        Self {
            slice: slice.into(),
            _dim: PhantomData,
        }
    }
}

/// A slice bound to a runtime dim name, for untyped tensors.
#[derive(Clone, Debug)]
pub struct StrSlice {
    pub(crate) name: String,
    pub(crate) slice: Slice,
}

impl StrSlice {
    pub fn new<N: Into<String>, S: Into<Slice>>(name: N, slice: S) -> Self {
        Self {
            name: name.into(),
            slice: slice.into(),
        }
    }
}

/// A named slice spec applicable to the typed dim list `S`.
///
/// Built by [`s!`] as a cons list `(DimSlice<A>, (DimSlice<B>, ()))`. Each
/// entry is checked against `S` at compile time via
/// [`Contains`](crate::typed::Contains); the axis itself is resolved at
/// runtime from the tensor's dim names.
#[diagnostic::on_unimplemented(
    message = "not a valid named slice spec for this tensor",
    label = "expected `s![Dim => range, …]` entries whose dims all appear in the tensor's dim list",
    note = "string-keyed entries (`s![\"M\" => …]`) only work on untyped tensors; typed tensors need dim markers (`s![M => …]`)"
)]
pub trait SliceSpec<S, Idx> {
    /// Overwrite the entries of `slices` this spec constrains.
    fn write(self, names: &[&'static str], slices: &mut [Slice]);
}

impl<S> SliceSpec<S, ()> for () {
    fn write(self, _names: &[&'static str], _slices: &mut [Slice]) {}
}

impl<S, D, I, Rest, IRest> SliceSpec<S, (I, IRest)> for (DimSlice<D>, Rest)
where
    D: DimName,
    S: crate::typed::Contains<D, I>,
    Rest: SliceSpec<S, IRest>,
{
    fn write(self, names: &[&'static str], slices: &mut [Slice]) {
        let axis = names
            .iter()
            .position(|&n| n == D::NAME)
            .unwrap_or_else(|| panic!("named-tensor: dim '{}' not found in {names:?}", D::NAME));
        slices[axis] = self.0.slice;
        self.1.write(names, slices);
    }
}

/// A named slice spec applicable to an untyped tensor: dims are resolved by
/// runtime name lookup, panicking on a missing dim. Both string-keyed and
/// typed entries qualify, so a typed `s![M => …]` spec can slice an untyped
/// tensor too.
#[diagnostic::on_unimplemented(
    message = "not a valid named slice spec",
    label = "expected `s![\"dim\" => range, …]` or `s![Dim => range, …]` entries"
)]
pub trait UntypedSliceSpec {
    /// Overwrite the entries of `slices` this spec constrains.
    fn write(self, names: &[String], slices: &mut [Slice]);
}

impl UntypedSliceSpec for () {
    fn write(self, _names: &[String], _slices: &mut [Slice]) {}
}

impl<Rest: UntypedSliceSpec> UntypedSliceSpec for (StrSlice, Rest) {
    fn write(self, names: &[String], slices: &mut [Slice]) {
        let axis = crate::untyped::axis_of(names, &self.0.name);
        slices[axis] = self.0.slice;
        self.1.write(names, slices);
    }
}

impl<D: DimName, Rest: UntypedSliceSpec> UntypedSliceSpec for (DimSlice<D>, Rest) {
    fn write(self, names: &[String], slices: &mut [Slice]) {
        let axis = crate::untyped::axis_of(names, D::NAME);
        slices[axis] = self.0.slice;
        self.1.write(names, slices);
    }
}

/// Named slice spec constructor, xarray-style: dims are addressed by name, in
/// any order, and unmentioned dims are kept whole.
///
/// * `s![Batch => 0..16, SeqLen => 5..10]` — typed spec (compile-checked
///   against the tensor's dim list)
/// * `s!["Batch" => 0..16]` — string-keyed spec for untyped tensors
///   (runtime-checked)
/// * `s![Batch => 0..16;2]` — per-dim step, like burn's `s!`
/// * `s![0..24;2]` — single bare extent, producing a plain [`Slice`] for
///   name-taking methods like `slice_by`
///
/// Ranges support negative indices counting from the end, as in burn's `s!`.
/// Positional multi-dim specs (`s![0..16, 5..10]`) are rejected: on a named
/// tensor every extent must be bound to a dim name.
#[macro_export]
macro_rules! s {
    // String-keyed entry (untyped tensors), with and without step. These
    // rules must precede the typed ones: `literal` also matches `expr`.
    ($name:literal => $range:expr; $step:expr $(, $($rest:tt)*)?) => {
        (
            $crate::StrSlice::new($name, $crate::Slice::from_range_stepped($range, $step as isize)),
            $crate::s!($($($rest)*)?),
        )
    };
    ($name:literal => $range:expr $(, $($rest:tt)*)?) => {
        (
            $crate::StrSlice::new($name, $range),
            $crate::s!($($($rest)*)?),
        )
    };
    // Typed entry (dim marker), with and without step.
    ($dim:expr => $range:expr; $step:expr $(, $($rest:tt)*)?) => {
        (
            $crate::DimSlice::new($dim, $crate::Slice::from_range_stepped($range, $step as isize)),
            $crate::s!($($($rest)*)?),
        )
    };
    ($dim:expr => $range:expr $(, $($rest:tt)*)?) => {
        (
            $crate::DimSlice::new($dim, $range),
            $crate::s!($($($rest)*)?),
        )
    };
    // Bare single extent (with/without step): a plain `Slice`, no dim binding.
    ($range:expr; $step:expr) => {
        $crate::Slice::from_range_stepped($range, $step as isize)
    };
    ($range:expr) => {
        $crate::Slice::from($range)
    };
    // Positional multi-dim specs are a named-tensor foot-gun; reject with help.
    ($range:expr; $step:expr, $($rest:tt)+) => {
        compile_error!(
            "positional multi-dim slicing is not supported on named tensors; \
             bind each extent to a dim name: s![Batch => 0..4, SeqLen => ..]"
        )
    };
    ($range:expr, $($rest:tt)+) => {
        compile_error!(
            "positional multi-dim slicing is not supported on named tensors; \
             bind each extent to a dim name: s![Batch => 0..4, SeqLen => ..]"
        )
    };
    // Recursion terminator (also makes `s![]` an empty spec: full view).
    () => { () };
}
