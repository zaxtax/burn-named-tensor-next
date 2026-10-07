use burn::prelude::*;
use burn::tensor::{Element, Slice, activation};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use super::dims::*;
use super::ops::NamedOut;

pub struct NamedTensor<S, const D: usize> {
    pub inner: Tensor<D>,
    pub names: Vec<&'static str>,
    _s: PhantomData<fn() -> S>,
}

impl<S: NameList + Rank, const D: usize> NamedTensor<S, D> {
    pub fn new(t: Tensor<D>) -> Self {
        debug_assert_eq!(D, S::RANK);
        Self {
            inner: t,
            names: S::names(),
            _s: PhantomData,
        }
    }

    /// Build from raw data, mirroring [`Tensor::from_data`]. Dim names come
    /// from the type `S`.
    pub fn from_data<T: Into<TensorData>>(data: T, device: &Device) -> Self {
        Self::new(Tensor::from_data(data, device))
    }

    /// Build from a flat `Vec` of elements and a shape.
    pub fn from_floats<E: Element>(
        data: Vec<E>,
        shape: impl Into<Shape>,
        device: &Device,
    ) -> Self {
        Self::new(Tensor::from_data(TensorData::new(data, shape), device))
    }

    pub fn into_inner(self) -> Tensor<D> {
        self.inner
    }
    pub fn shape(&self) -> burn::tensor::Shape {
        self.inner.shape()
    }
    pub fn dim_names(&self) -> &[&'static str] {
        &self.names
    }
    pub fn dims_str(&self) -> String {
        format!("({})", self.names.join(","))
    }

    /// Mean-reduce over named dims `Ks`.
    pub fn mean<Ks, Out, Idx, const D_OUT: usize>(self) -> <Out as NamedOut<D_OUT>>::Out
    where
        Ks: NameList,
        S: RemoveAll<Ks, Idx, Output = Out>,
        Out: NamedOut<D_OUT>,
    {
        super::ops::mean::<Ks, Out, S, Idx, D, D_OUT>(self)
    }

    /// Drop to an untyped [`crate::untyped::NamedTensor`] for runtime-checked operations.
    pub fn untyped(self) -> crate::untyped::NamedTensor<D> {
        let names: [String; D] = std::array::from_fn(|i| self.names[i].to_string());
        crate::untyped::NamedTensor::from_parts(names, self.inner)
    }

    /// Returns a view restricted to the extents of a named slice spec built
    /// with [`s!`](crate::s): dims in any order, unmentioned dims kept whole.
    /// Rank and dim names are unchanged; every dim in the spec is checked at
    /// compile time.
    pub fn slice<Spec, Idx>(self, spec: Spec) -> Self
    where
        Spec: crate::slice::SliceSpec<S, Idx>,
    {
        let mut slices = vec![Slice::full(); D];
        spec.write(&self.names, &mut slices);
        NamedTensor::new(self.inner.slice(&slices))
    }

    /// Slice along the single dim `dim`, keeping rank. Accepts anything
    /// convertible to a [`Slice`], including `s![0..24;2]` for stepped
    /// extents.
    pub fn slice_by<Dm, I, Sl>(self, _dim: Dm, slice: Sl) -> Self
    where
        Dm: DimName,
        S: Contains<Dm, I>,
        Sl: Into<Slice>,
    {
        let axis = find_axis(&self.names, Dm::NAME);
        NamedTensor::new(self.inner.slice_dim(axis, slice))
    }

    /// Assigns `values` to the region selected by a named slice spec and
    /// returns the updated tensor.
    pub fn slice_assign<Spec, Idx>(self, spec: Spec, values: Self) -> Self
    where
        Spec: crate::slice::SliceSpec<S, Idx>,
    {
        let mut slices = vec![Slice::full(); D];
        spec.write(&self.names, &mut slices);
        NamedTensor::new(self.inner.slice_assign(&slices, values.inner))
    }

    /// Fills the region selected by a named slice spec with `value` and
    /// returns the updated tensor.
    pub fn slice_fill<Spec, Idx, E>(self, spec: Spec, value: E) -> Self
    where
        Spec: crate::slice::SliceSpec<S, Idx>,
        E: burn::tensor::Element,
    {
        let mut slices = vec![Slice::full(); D];
        spec.write(&self.names, &mut slices);
        NamedTensor::new(self.inner.slice_fill(&slices, value))
    }

    /// Selects a single index along dim `Dm`, removing that dim from the
    /// result (xarray's `isel` semantics for integer indexers). Negative
    /// indices count from the end.
    pub fn isel_by<Dm, I, Out, const D_OUT: usize>(
        self,
        _dim: Dm,
        index: isize,
    ) -> NamedTensor<Out, D_OUT>
    where
        Dm: DimName,
        S: Contains<Dm, I> + Remove<Dm, I, Output = Out>,
        Out: NameList + Rank,
    {
        let axis = find_axis(&self.names, Dm::NAME);
        NamedTensor::new(self.inner.slice_dim(axis, index).squeeze_dim(axis))
    }

    /// Removes dim `Dm` from the tensor. Panics if its size is not 1; use
    /// [`isel_by`](Self::isel_by) to pick an index along a larger dim.
    pub fn squeeze_dim<Dm, I, Out, const D_OUT: usize>(self, _dim: Dm) -> NamedTensor<Out, D_OUT>
    where
        Dm: DimName,
        S: Contains<Dm, I> + Remove<Dm, I, Output = Out>,
        Out: NameList + Rank,
    {
        let axis = find_axis(&self.names, Dm::NAME);
        NamedTensor::new(self.inner.squeeze_dim(axis))
    }

    /// Removes the dims listed in `Ks`, e.g. `t.squeeze::<dims![M, K], _, _, 1>()`.
    /// Panics if any of them has a size other than 1.
    ///
    /// Unlike burn's `squeeze`, the dims to drop are named explicitly rather
    /// than inferred from runtime sizes: which dims disappear must be known
    /// at compile time, since they are removed from the type.
    pub fn squeeze<Ks, Out, Idx, const D_OUT: usize>(self) -> NamedTensor<Out, D_OUT>
    where
        Ks: NameList,
        S: RemoveAll<Ks, Idx, Output = Out>,
        Out: NameList + Rank,
    {
        let axes: Vec<isize> = Ks::names()
            .iter()
            .map(|k| find_axis(&self.names, k) as isize)
            .collect();
        NamedTensor::new(self.inner.squeeze_dims(&axes))
    }

    /// Align to the target dim list `Out`, permuting axes and adding size-1
    /// dims for any target dim not present in `self`. Every dim of `self` must
    /// appear in `Out` (checked at compile time); `Out` may contain extra dims.
    ///
    /// ```ignore
    /// let y: NamedTensor<dims![N, H, M], 3> = x.align_to();
    /// ```
    pub fn align_to<Out, Idx, const D_OUT: usize>(self) -> NamedTensor<Out, D_OUT>
    where
        S: Subset<Out, Idx> + NameList + Rank,
        Out: NameList + Rank,
    {
        super::ops::align_to::<Out, S, Idx, D, D_OUT>(self)
    }

    /// Align to the dim list of `other`, permuting axes and adding size-1 dims
    /// for any of `other`'s dims not present in `self`. `other` is borrowed only
    /// for its type.
    pub fn align_as<SR, Idx, const DR: usize>(
        self,
        other: &NamedTensor<SR, DR>,
    ) -> NamedTensor<SR, DR>
    where
        S: Subset<SR, Idx> + NameList + Rank,
        SR: NameList + Rank,
    {
        super::ops::align_as::<S, SR, Idx, D, DR>(self, other)
    }

    // ── Unary ops ──

    /// Element-wise rectified linear unit: `max(0, x)`.
    pub fn relu(self) -> Self {
        NamedTensor::new(activation::relu(self.inner))
    }
    /// Element-wise Gaussian Error Linear Unit.
    pub fn gelu(self) -> Self {
        NamedTensor::new(activation::gelu(self.inner))
    }
    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    pub fn sigmoid(self) -> Self {
        NamedTensor::new(activation::sigmoid(self.inner))
    }
    /// Element-wise SiLU / swish: `x * sigmoid(x)`.
    pub fn silu(self) -> Self {
        NamedTensor::new(activation::silu(self.inner))
    }
    /// Softmax along the named dim `Dm`.
    pub fn softmax<Dm, I>(self, _dim: Dm) -> Self
    where
        Dm: DimName,
        S: Contains<Dm, I>,
    {
        let axis = find_axis(&self.names, Dm::NAME);
        NamedTensor::new(activation::softmax(self.inner, axis))
    }
    /// Log-softmax along the named dim `Dm`.
    pub fn log_softmax<Dm, I>(self, _dim: Dm) -> Self
    where
        Dm: DimName,
        S: Contains<Dm, I>,
    {
        let axis = find_axis(&self.names, Dm::NAME);
        NamedTensor::new(activation::log_softmax(self.inner, axis))
    }

    /// Element-wise exponential: `e^x`.
    pub fn exp(self) -> Self {
        NamedTensor::new(self.inner.exp())
    }
    /// Element-wise natural logarithm: `ln(x)`.
    pub fn log(self) -> Self {
        NamedTensor::new(self.inner.log())
    }
    /// Element-wise `ln(x + 1)`.
    pub fn log1p(self) -> Self {
        NamedTensor::new(self.inner.log1p())
    }
    /// Element-wise square root.
    pub fn sqrt(self) -> Self {
        NamedTensor::new(self.inner.sqrt())
    }
    /// Element-wise square: `x * x`.
    pub fn square(self) -> Self {
        NamedTensor::new(self.inner.square())
    }
    /// Element-wise reciprocal: `1 / x`.
    pub fn recip(self) -> Self {
        NamedTensor::new(self.inner.recip())
    }

    /// Element-wise absolute value.
    pub fn abs(self) -> Self {
        NamedTensor::new(self.inner.abs())
    }
    /// Element-wise negation: `-x`.
    pub fn neg(self) -> Self {
        NamedTensor::new(self.inner.neg())
    }
    /// Element-wise sign: `1`, `0`, or `-1`.
    pub fn sign(self) -> Self {
        NamedTensor::new(self.inner.sign())
    }

    /// Element-wise sine.
    pub fn sin(self) -> Self {
        NamedTensor::new(self.inner.sin())
    }
    /// Element-wise cosine.
    pub fn cos(self) -> Self {
        NamedTensor::new(self.inner.cos())
    }
    /// Element-wise tangent.
    pub fn tan(self) -> Self {
        NamedTensor::new(self.inner.tan())
    }
    /// Element-wise hyperbolic sine.
    pub fn sinh(self) -> Self {
        NamedTensor::new(self.inner.sinh())
    }
    /// Element-wise hyperbolic cosine.
    pub fn cosh(self) -> Self {
        NamedTensor::new(self.inner.cosh())
    }
    /// Element-wise hyperbolic tangent.
    pub fn tanh(self) -> Self {
        NamedTensor::new(self.inner.tanh())
    }

    /// Element-wise error function.
    pub fn erf(self) -> Self {
        NamedTensor::new(self.inner.erf())
    }
    /// Element-wise round to nearest (half-to-even).
    pub fn round(self) -> Self {
        NamedTensor::new(self.inner.round())
    }
    /// Element-wise floor.
    pub fn floor(self) -> Self {
        NamedTensor::new(self.inner.floor())
    }
    /// Element-wise ceil.
    pub fn ceil(self) -> Self {
        NamedTensor::new(self.inner.ceil())
    }

    /// Clamp all elements to `[min, max]`.
    pub fn clamp<E: burn::tensor::ElementConversion>(self, min: E, max: E) -> Self {
        NamedTensor::new(self.inner.clamp(min, max))
    }
    /// Clamp all elements to `[min, +∞)`.
    pub fn clamp_min<E: burn::tensor::ElementConversion>(self, min: E) -> Self {
        NamedTensor::new(self.inner.clamp_min(min))
    }
    /// Clamp all elements to `(-∞, max]`.
    pub fn clamp_max<E: burn::tensor::ElementConversion>(self, max: E) -> Self {
        NamedTensor::new(self.inner.clamp_max(max))
    }

    /// Reverse along the named dim `Dm`.
    pub fn flip<Dm, I>(self, _dim: Dm) -> Self
    where
        Dm: DimName,
        S: Contains<Dm, I>,
    {
        let axis = find_axis(&self.names, Dm::NAME);
        NamedTensor::new(self.inner.flip([axis as isize]))
    }

    /// Cumulative sum over the named dims in `Ks`, keeping rank and dim list.
    /// Over a single dim this is a running sum; over several it is the
    /// N-dimensional prefix sum (the "corner" sum — order-independent because
    /// cumsum is linear), mirroring xarray's `cumsum(dim=[...])`.
    ///
    /// ```ignore
    /// let a: NamedTensor<dims![N], 1>    = t.cumsum::<dims![N], _>();
    /// let b: NamedTensor<dims![M, N], 2> = t.cumsum::<dims![M, N], _>();
    /// ```
    pub fn cumsum<Ks, Idx>(self) -> Self
    where
        Ks: NameList,
        Ks: Subset<S, Idx>,
    {
        let mut inner = self.inner;
        for k in Ks::names() {
            inner = inner.cumsum(find_axis(&self.names, k));
        }
        NamedTensor::new(inner)
    }

    /// Private single-dim variant of [`cumsum`], kept for call sites that
    /// hold a concrete dim marker `Dm` rather than a dim list.
    #[allow(dead_code)]
    fn cumsum_dim<Dm, I>(self, _dim: Dm) -> Self
    where
        Dm: DimName,
        S: Contains<Dm, I>,
    {
        let axis = find_axis(&self.names, Dm::NAME);
        NamedTensor::new(self.inner.cumsum(axis))
    }
}

impl<S, const D: usize> Clone for NamedTensor<S, D>
where
    Tensor<D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            names: self.names.clone(),
            _s: PhantomData,
        }
    }
}
impl<S, const D: usize> std::fmt::Debug for NamedTensor<S, D>
where
    Tensor<D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}
impl<S, const D: usize> std::fmt::Display for NamedTensor<S, D>
where
    Tensor<D>: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}

// Operators use lhs as the output type: all rhs dims must be present in lhs.
// For true union broadcasting (where the output has dims from both sides),
// use the free functions `add`, `sub`, `mul`, `div` with an explicit output type.
macro_rules! impl_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<SL, SR, const DL: usize, const DR: usize> $trait<NamedTensor<SR, DR>>
            for NamedTensor<SL, DL>
        where
            SL: NameList + Rank,
            SR: NameList + Rank,
        {
            type Output = NamedTensor<SL, DL>;

            fn $method(self, rhs: NamedTensor<SR, DR>) -> Self::Output {
                let r = align_to_impl(rhs.inner, &rhs.names, &self.names);
                NamedTensor::new(self.inner $op r)
            }
        }
    };
}

impl_op!(Add, add, +);
impl_op!(Sub, sub, -);
impl_op!(Mul, mul, *);
impl_op!(Div, div, /);
