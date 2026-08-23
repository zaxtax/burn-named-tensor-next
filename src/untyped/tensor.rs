use burn::prelude::*;
use burn::tensor::{Element, Shape, Slice, activation};
use std::ops::{Add, Div, Mul, Sub};

use super::ops::{align, axis_of, perm_of, permute_by, to_array};

pub struct NamedTensor<B: Backend, const D: usize> {
    pub inner: Tensor<B, D>,
    pub names: [String; D],
}

impl<B: Backend, const D: usize> NamedTensor<B, D> {
    pub fn new(names: [&str; D], inner: Tensor<B, D>) -> Self {
        Self {
            inner,
            names: names.map(String::from),
        }
    }
    pub fn from_parts(names: [String; D], inner: Tensor<B, D>) -> Self {
        Self { inner, names }
    }

    /// Build from raw data, mirroring [`Tensor::from_data`].
    pub fn from_data<T: Into<TensorData>>(names: [&str; D], data: T, device: &B::Device) -> Self {
        Self::new(names, Tensor::from_data(data, device))
    }

    /// Build from a flat `Vec` of elements and a shape.
    pub fn from_floats<E: Element>(
        names: [&str; D],
        data: Vec<E>,
        shape: impl Into<Shape>,
        device: &B::Device,
    ) -> Self {
        Self::new(names, Tensor::from_data(TensorData::new(data, shape), device))
    }

    pub fn into_inner(self) -> Tensor<B, D> {
        self.inner
    }
    pub fn shape(&self) -> Shape {
        self.inner.shape()
    }
    pub fn names(&self) -> &[String; D] {
        &self.names
    }

    /// Mean-reduce over the given dims.
    pub fn mean<C: super::ops::IntoContract, const D_OUT: usize>(
        self,
        dims: C,
    ) -> NamedTensor<B, D_OUT> {
        let contract = dims.into_contract();
        assert_eq!(
            D_OUT + contract.len(),
            D,
            "mean: D_OUT ({D_OUT}) must equal D ({D}) minus contracted dims ({})",
            contract.len()
        );
        let mut inner = self.inner;
        for d in &contract {
            let axis = axis_of(&self.names, d);
            inner = inner.mean_dim(axis);
        }
        let out_names: Vec<String> = self
            .names
            .iter()
            .filter(|n| !contract.contains(n))
            .cloned()
            .collect();
        let shape = inner.shape().to_vec();
        let kept: Vec<usize> = self
            .names
            .iter()
            .enumerate()
            .filter(|(_, n)| !contract.contains(n))
            .map(|(i, _)| shape[i])
            .collect();
        let out_shape: [usize; D_OUT] = std::array::from_fn(|i| kept[i]);
        let prod: usize = out_shape.iter().product::<usize>().max(1);
        let flat: Tensor<B, 1> = inner.reshape([prod]);
        let result: Tensor<B, D_OUT> = flat.reshape(out_shape);
        NamedTensor::from_parts(to_array(out_names), result)
    }

    /// Returns a view restricted to the extents of a named slice spec built
    /// with [`s!`](crate::s): dims in any order, unmentioned dims kept whole.
    /// Both string-keyed (`s!["M" => 0..2]`) and typed (`s![M => 0..2]`)
    /// specs work; panics if a spec dim is not present.
    pub fn slice<Spec: crate::slice::UntypedSliceSpec>(self, spec: Spec) -> Self {
        let mut slices = vec![Slice::full(); D];
        spec.write(&self.names, &mut slices);
        Self::from_parts(self.names, self.inner.slice(&slices))
    }

    /// Slice along the single dim `dim`, keeping rank. Accepts anything
    /// convertible to a [`Slice`], including `s![0..24;2]` for stepped
    /// extents.
    pub fn slice_by<Sl: Into<Slice>>(self, dim: &str, slice: Sl) -> Self {
        let axis = axis_of(&self.names, dim);
        Self::from_parts(self.names, self.inner.slice_dim(axis, slice))
    }

    /// Assigns `values` to the region selected by a named slice spec and
    /// returns the updated tensor. `values` is aligned by dim name first, so
    /// its axes may be in a different order.
    pub fn slice_assign<Spec: crate::slice::UntypedSliceSpec>(
        self,
        spec: Spec,
        values: NamedTensor<B, D>,
    ) -> Self {
        let mut slices = vec![Slice::full(); D];
        spec.write(&self.names, &mut slices);
        let v = permute_by(values.inner, &perm_of(&values.names, &self.names));
        Self::from_parts(self.names, self.inner.slice_assign(&slices, v))
    }

    /// Fills the region selected by a named slice spec with `value` and
    /// returns the updated tensor.
    pub fn slice_fill<Spec, E>(self, spec: Spec, value: E) -> Self
    where
        Spec: crate::slice::UntypedSliceSpec,
        E: burn::tensor::ElementConversion,
    {
        let mut slices = vec![Slice::full(); D];
        spec.write(&self.names, &mut slices);
        Self::from_parts(self.names, self.inner.slice_fill(&slices, value))
    }

    /// Selects a single index along `dim`, removing that dim from the result
    /// (xarray's `isel` semantics for integer indexers). Negative indices
    /// count from the end.
    pub fn isel_by<const D_OUT: usize>(self, dim: &str, index: isize) -> NamedTensor<B, D_OUT> {
        assert_eq!(D_OUT + 1, D, "isel_by: D_OUT must equal D-1");
        let axis = axis_of(&self.names, dim);
        let mut names = self.names.to_vec();
        names.remove(axis);
        NamedTensor::from_parts(
            to_array(names),
            self.inner.slice_dim(axis, index).squeeze_dim(axis),
        )
    }

    /// Removes `dim` from the tensor. Panics if it is missing or its size is
    /// not 1; use [`isel_by`](Self::isel_by) to pick an index along a larger
    /// dim.
    pub fn squeeze_dim<const D_OUT: usize>(self, dim: &str) -> NamedTensor<B, D_OUT> {
        assert_eq!(D_OUT + 1, D, "squeeze_dim: D_OUT must equal D-1");
        let axis = axis_of(&self.names, dim);
        let mut names = self.names.to_vec();
        names.remove(axis);
        NamedTensor::from_parts(to_array(names), self.inner.squeeze_dim(axis))
    }

    /// Removes every dim of size 1, like burn's `squeeze`. Panics if the
    /// number of remaining dims doesn't match `D_OUT`.
    pub fn squeeze<const D_OUT: usize>(self) -> NamedTensor<B, D_OUT> {
        let shape = self.inner.shape().to_vec();
        let names: Vec<String> = self
            .names
            .iter()
            .zip(&shape)
            .filter(|(_, size)| **size != 1)
            .map(|(n, _)| n.clone())
            .collect();
        assert_eq!(
            names.len(),
            D_OUT,
            "squeeze: {} dims of size > 1 in {:?}, expected D_OUT={D_OUT}",
            names.len(),
            self.names,
        );
        NamedTensor::from_parts(to_array(names), self.inner.squeeze())
    }

    /// Convert to a typed [`crate::typed::NamedTensor`], permuting axes to match
    /// the target dim order. Panics if the name sets don't match.
    pub fn to_named<S: crate::typed::NameList + crate::typed::Rank>(
        self,
    ) -> crate::typed::NamedTensor<B, S, D> {
        let target = S::names();
        let from: Vec<String> = self.names.to_vec();
        let to: Vec<String> = target.iter().map(|s| s.to_string()).collect();
        let inner = permute_by(self.inner, &perm_of(&from, &to));
        crate::typed::NamedTensor::new(inner)
    }

    // ── Unary ops ──

    /// Element-wise rectified linear unit: `max(0, x)`.
    pub fn relu(self) -> Self {
        Self::from_parts(self.names, activation::relu(self.inner))
    }
    /// Element-wise Gaussian Error Linear Unit.
    pub fn gelu(self) -> Self {
        Self::from_parts(self.names, activation::gelu(self.inner))
    }
    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    pub fn sigmoid(self) -> Self {
        Self::from_parts(self.names, activation::sigmoid(self.inner))
    }
    /// Element-wise SiLU / swish: `x * sigmoid(x)`.
    pub fn silu(self) -> Self {
        Self::from_parts(self.names, activation::silu(self.inner))
    }
    /// Softmax along the named dim `dim`.
    pub fn softmax(self, dim: &str) -> Self {
        let axis = axis_of(&self.names, dim);
        Self::from_parts(self.names, activation::softmax(self.inner, axis))
    }
    /// Log-softmax along the named dim `dim`.
    pub fn log_softmax(self, dim: &str) -> Self {
        let axis = axis_of(&self.names, dim);
        Self::from_parts(self.names, activation::log_softmax(self.inner, axis))
    }

    /// Element-wise exponential: `e^x`.
    pub fn exp(self) -> Self {
        Self::from_parts(self.names, self.inner.exp())
    }
    /// Element-wise natural logarithm: `ln(x)`.
    pub fn log(self) -> Self {
        Self::from_parts(self.names, self.inner.log())
    }
    /// Element-wise `ln(x + 1)`.
    pub fn log1p(self) -> Self {
        Self::from_parts(self.names, self.inner.log1p())
    }
    /// Element-wise square root.
    pub fn sqrt(self) -> Self {
        Self::from_parts(self.names, self.inner.sqrt())
    }
    /// Element-wise square: `x * x`.
    pub fn square(self) -> Self {
        Self::from_parts(self.names, self.inner.square())
    }
    /// Element-wise reciprocal: `1 / x`.
    pub fn recip(self) -> Self {
        Self::from_parts(self.names, self.inner.recip())
    }

    /// Element-wise absolute value.
    pub fn abs(self) -> Self {
        Self::from_parts(self.names, self.inner.abs())
    }
    /// Element-wise negation: `-x`.
    pub fn neg(self) -> Self {
        Self::from_parts(self.names, self.inner.neg())
    }
    /// Element-wise sign: `1`, `0`, or `-1`.
    pub fn sign(self) -> Self {
        Self::from_parts(self.names, self.inner.sign())
    }

    /// Element-wise sine.
    pub fn sin(self) -> Self {
        Self::from_parts(self.names, self.inner.sin())
    }
    /// Element-wise cosine.
    pub fn cos(self) -> Self {
        Self::from_parts(self.names, self.inner.cos())
    }
    /// Element-wise tangent.
    pub fn tan(self) -> Self {
        Self::from_parts(self.names, self.inner.tan())
    }
    /// Element-wise hyperbolic sine.
    pub fn sinh(self) -> Self {
        Self::from_parts(self.names, self.inner.sinh())
    }
    /// Element-wise hyperbolic cosine.
    pub fn cosh(self) -> Self {
        Self::from_parts(self.names, self.inner.cosh())
    }
    /// Element-wise hyperbolic tangent.
    pub fn tanh(self) -> Self {
        Self::from_parts(self.names, self.inner.tanh())
    }

    /// Element-wise error function.
    pub fn erf(self) -> Self {
        Self::from_parts(self.names, self.inner.erf())
    }
    /// Element-wise round to nearest (half-to-even).
    pub fn round(self) -> Self {
        Self::from_parts(self.names, self.inner.round())
    }
    /// Element-wise floor.
    pub fn floor(self) -> Self {
        Self::from_parts(self.names, self.inner.floor())
    }
    /// Element-wise ceil.
    pub fn ceil(self) -> Self {
        Self::from_parts(self.names, self.inner.ceil())
    }

    /// Clamp all elements to `[min, max]`.
    pub fn clamp<E: burn::tensor::ElementConversion>(self, min: E, max: E) -> Self {
        Self::from_parts(self.names, self.inner.clamp(min, max))
    }
    /// Clamp all elements to `[min, +∞)`.
    pub fn clamp_min<E: burn::tensor::ElementConversion>(self, min: E) -> Self {
        Self::from_parts(self.names, self.inner.clamp_min(min))
    }
    /// Clamp all elements to `(-∞, max]`.
    pub fn clamp_max<E: burn::tensor::ElementConversion>(self, max: E) -> Self {
        Self::from_parts(self.names, self.inner.clamp_max(max))
    }

    /// Reverse along the named dim `dim`.
    pub fn flip(self, dim: &str) -> Self {
        let axis = axis_of(&self.names, dim);
        Self::from_parts(self.names, self.inner.flip([axis as isize]))
    }
}

impl<B: Backend, const D: usize> Clone for NamedTensor<B, D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            names: self.names.clone(),
        }
    }
}
impl<B: Backend, const D: usize> std::fmt::Debug for NamedTensor<B, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedTensor")
            .field("names", &self.names)
            .field("inner", &self.inner)
            .finish()
    }
}
impl<B: Backend, const D: usize> std::fmt::Display for NamedTensor<B, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}", self.names, self.inner)
    }
}

// Operators use lhs as the output type: all rhs dims must be present in lhs.
// For true union broadcasting (where the output has dims from both sides),
// use the free functions `add`, `sub`, `mul`, `div` with an explicit output type.
macro_rules! impl_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<B: Backend, const DL: usize, const DR: usize> $trait<NamedTensor<B, DR>>
            for NamedTensor<B, DL>
        {
            type Output = NamedTensor<B, DL>;

            fn $method(self, rhs: NamedTensor<B, DR>) -> Self::Output {
                let r = align::<B, DR, DL>(rhs.inner, &rhs.names, &self.names);
                NamedTensor::from_parts(self.names, self.inner $op r)
            }
        }
    };
}

impl_op!(Add, add, +);
impl_op!(Sub, sub, -);
impl_op!(Mul, mul, *);
impl_op!(Div, div, /);
