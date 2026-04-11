//! Named-dimension tensors backed by [`burn`] on stable Rust.

use burn::prelude::*;
use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul, Div};

pub trait DimName {
    const NAME: &'static str;
}

#[macro_export]
macro_rules! dim {
    ($($name:ident),+ $(,)?) => {
        $(
            pub struct $name;
            impl $crate::DimName for $name {
                const NAME: &'static str = stringify!($name);
            }
        )+
    };
}

pub struct Here;
pub struct There<I>(PhantomData<I>);

/// Marker: this dim is shared with the other tensor (will be contracted).
pub struct IsShared;
/// Marker: this dim is exclusive to this tensor (will be kept).
pub struct IsExclusive;

pub struct DNil;
pub struct DCons<H, T>(PhantomData<fn() -> (H, T)>);

#[macro_export]
macro_rules! dims {
    ()                  => { $crate::DNil };
    ($h:ty)             => { $crate::DCons<$h, $crate::DNil> };
    ($h:ty, $($t:ty),+) => { $crate::DCons<$h, $crate::dims![$($t),+]> };
}

pub trait Rank {
    const RANK: usize;
}
impl Rank for DNil {
    const RANK: usize = 0;
}
impl<H, T: Rank> Rank for DCons<H, T> {
    const RANK: usize = 1 + T::RANK;
}

pub trait NameList {
    fn names() -> Vec<&'static str>;
}
impl NameList for DNil {
    fn names() -> Vec<&'static str> {
        vec![]
    }
}
impl<H: DimName, T: NameList> NameList for DCons<H, T> {
    fn names() -> Vec<&'static str> {
        let mut v = vec![H::NAME];
        v.extend(T::names());
        v
    }
}

#[diagnostic::on_unimplemented(
    message = "dim `{D}` is not present in this tensor's dimension list",
    label = "dim `{D}` missing here",
    note = "double-check the dim markers on this `NamedTensor` — `{D}` must appear in its `dims![…]` list"
)]
pub trait Contains<D, Idx> {}
impl<D, T> Contains<D, Here> for DCons<D, T> {}
impl<H, D, T, I> Contains<D, There<I>> for DCons<H, T> where T: Contains<D, I> {}

#[diagnostic::on_unimplemented(
    message = "cannot remove dim `{D}` — it is not present in the dimension list",
    note = "this usually appears alongside a `Contains<{D}, _>` error on the same line; fixing the missing dim resolves both"
)]
pub trait Remove<D, Idx> {
    type Output;
}
impl<D, T> Remove<D, Here> for DCons<D, T> {
    type Output = T;
}
impl<H, D, T, I> Remove<D, There<I>> for DCons<H, T>
where
    T: Remove<D, I>,
{
    type Output = DCons<H, <T as Remove<D, I>>::Output>;
}

pub trait Append<D> {
    type Output;
}
impl<D> Append<D> for DNil {
    type Output = DCons<D, DNil>;
}
impl<H, T: Append<D>, D> Append<D> for DCons<H, T> {
    type Output = DCons<H, <T as Append<D>>::Output>;
}

#[diagnostic::on_unimplemented(
    message = "dim list `{Ks}` is not fully contained in `{Self}`",
    note = "every dim in `{Ks}` must appear in `{Self}` so it can be removed"
)]
pub trait RemoveAll<Ks, Idx> {
    type Output;
}
impl<S> RemoveAll<DNil, ()> for S {
    type Output = S;
}
impl<S, KH, KT, IH, IT> RemoveAll<DCons<KH, KT>, (IH, IT)> for S
where
    S: Remove<KH, IH>,
    <S as Remove<KH, IH>>::Output: RemoveAll<KT, IT>,
{
    type Output = <<S as Remove<KH, IH>>::Output as RemoveAll<KT, IT>>::Output;
}

/// Partitions `Self` relative to `Other`, using `Out` (the expected output dim
/// list) to disambiguate: a **shared** dim appears in `Other` (and is
/// contracted away), while an **exclusive** dim appears in `Out` (and is kept).
///
/// The compiler can always pick exactly one branch because:
/// - A shared dim is in `Other` but *not* in `Out` → only `IsShared` applies.
/// - An exclusive dim is in `Out` but *not* in `Other` → only `IsExclusive` applies.
/// - If a dim is in *both* `Other` and `Out`, the user's output annotation is
///   wrong (a contracted dim shouldn't survive); the compiler reports ambiguity.
/// - If a dim is in *neither*, the output is missing a required dim; the
///   compiler reports an unsatisfied bound.
#[diagnostic::on_unimplemented(
    message = "cannot partition `{Self}` into shared/exclusive dims given `{Other}` and output `{Out}`",
    note = "each dim in `{Self}` must be shared (in `{Other}`, contracted) or exclusive (in `{Out}`, kept) — but not both and not neither"
)]
pub trait Exclusive<Other, Out, Idx> {
    type Output;
}
impl<Other, Out> Exclusive<Other, Out, ()> for DNil {
    type Output = DNil;
}
/// Dim `H` is shared with `Other` — contracted away, not in output.
impl<H, T, Other, Out, IH, IT> Exclusive<Other, Out, (IsShared, IH, IT)> for DCons<H, T>
where
    Other: Contains<H, IH>,
    T: Exclusive<Other, Out, IT>,
{
    type Output = <T as Exclusive<Other, Out, IT>>::Output;
}
/// Dim `H` is exclusive to `Self` — kept in output.
impl<H, T, Other, Out, IO, IT> Exclusive<Other, Out, (IsExclusive, IO, IT)> for DCons<H, T>
where
    Out: Contains<H, IO>,
    T: Exclusive<Other, Out, IT>,
{
    type Output = DCons<H, <T as Exclusive<Other, Out, IT>>::Output>;
}

#[diagnostic::on_unimplemented(
    message = "dimension list `{Self}` is not a subset of `{Out}`",
    label = "some dim in `{Self}` is missing from `{Out}`",
    note = "every dim in `{Self}` must also appear in `{Out}` (order doesn't matter, but the set must match or be contained)"
)]
pub trait Subset<Out, Idx> {}
impl<Out> Subset<Out, ()> for DNil {}
impl<H, T, Out, IC, IT> Subset<Out, (IC, IT)> for DCons<H, T>
where
    Out: Contains<H, IC>,
    T: Subset<Out, IT>,
{
}

#[diagnostic::on_unimplemented(
    message = "output dims `{Self}` are not a valid union of `{SL}` and `{SR}`",
    label = "output `{Self}` must contain every dim from both inputs",
    note = "the annotated output dimension list must be a superset of both `{SL}` and `{SR}` — add any missing dim from either side to your output annotation"
)]
pub trait IsUnionOf<SL, SR, Idx> {}
impl<Out, SL, SR, LIdx, RIdx> IsUnionOf<SL, SR, (LIdx, RIdx)> for Out
where
    SL: Subset<Out, LIdx>,
    SR: Subset<Out, RIdx>,
{
}

#[diagnostic::on_unimplemented(
    message = "cannot rename dim `{Old}` to `{New}` — `{Old}` is not present in the dimension list",
    note = "`rename` requires the old dim to exist. If you're trying to introduce a brand-new dim, `rename` is not the right tool — construct a fresh `NamedTensor` with the desired marker instead"
)]
pub trait ReplaceFirst<Old, New, Idx> {
    type Output;
}
impl<Old, New, T> ReplaceFirst<Old, New, Here> for DCons<Old, T> {
    type Output = DCons<New, T>;
}
impl<H, Old, New, T, I> ReplaceFirst<Old, New, There<I>> for DCons<H, T>
where
    T: ReplaceFirst<Old, New, I>,
{
    type Output = DCons<H, <T as ReplaceFirst<Old, New, I>>::Output>;
}

fn find_axis(list: &[&'static str], name: &'static str) -> usize {
    list.iter()
        .position(|&n| n == name)
        .unwrap_or_else(|| panic!("named-tensor: dim '{name}' not found in {list:?}"))
}

fn build_perm(from: &[&'static str], to: &[&'static str]) -> Vec<usize> {
    to.iter().map(|name| find_axis(from, name)).collect()
}

fn is_identity(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

fn permute_if_needed<B: Backend, const D: usize>(t: Tensor<B, D>, perm: &[usize]) -> Tensor<B, D> {
    if is_identity(perm) {
        return t;
    }
    let arr: [isize; D] = std::array::from_fn(|i| perm[i] as isize);
    t.permute(arr)
}

fn align_to<B: Backend, const D_IN: usize, const D_OUT: usize>(
    t: Tensor<B, D_IN>,
    operand_names: &[&'static str],
    target_names: &[&'static str],
) -> Tensor<B, D_OUT> {
    let missing: Vec<isize> = (0..D_OUT as isize)
        .filter(|&i| !operand_names.contains(&target_names[i as usize]))
        .collect();

    let expanded: Tensor<B, D_OUT> = t.unsqueeze_dims(&missing);

    let mut src = operand_names.iter().copied();
    let current: Vec<&'static str> = (0..D_OUT)
        .map(|i| {
            if missing.contains(&(i as isize)) {
                target_names[i]
            } else {
                src.next().unwrap()
            }
        })
        .collect();

    permute_if_needed(expanded, &build_perm(&current, target_names))
}

pub struct NamedTensor<B: Backend, S, const D: usize> {
    pub inner: Tensor<B, D>,
    pub names: Vec<&'static str>,
    _s: PhantomData<fn() -> S>,
}

impl<B: Backend, S: NameList + Rank, const D: usize> NamedTensor<B, S, D> {
    pub fn new(t: Tensor<B, D>) -> Self {
        debug_assert_eq!(D, S::RANK);
        Self {
            inner: t,
            names: S::names(),
            _s: PhantomData,
        }
    }
    pub fn into_inner(self) -> Tensor<B, D> {
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
    pub fn mean<Ks, Out, Idx, const D_OUT: usize>(self) -> <Out as NamedOut<B, D_OUT>>::Out
    where
        Ks: NameList,
        S: RemoveAll<Ks, Idx, Output = Out>,
        Out: NamedOut<B, D_OUT>,
    {
        mean::<B, Ks, Out, S, Idx, D, D_OUT>(self)
    }

    /// Drop to an untyped [`crate::untyped::NamedTensor`] for runtime-checked operations.
    pub fn untyped(self) -> crate::untyped::NamedTensor<B, D> {
        let names: [String; D] = std::array::from_fn(|i| self.names[i].to_string());
        crate::untyped::NamedTensor::from_parts(names, self.inner)
    }
}

impl<B: Backend, S, const D: usize> Clone for NamedTensor<B, S, D>
where
    Tensor<B, D>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            names: self.names.clone(),
            _s: PhantomData,
        }
    }
}
impl<B: Backend, S, const D: usize> std::fmt::Debug for NamedTensor<B, S, D>
where
    Tensor<B, D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}
impl<B: Backend, S, const D: usize> std::fmt::Display for NamedTensor<B, S, D>
where
    Tensor<B, D>: std::fmt::Display,
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
        impl<B, SL, SR, const DL: usize, const DR: usize> $trait<NamedTensor<B, SR, DR>>
            for NamedTensor<B, SL, DL>
        where
            B: Backend,
            SL: NameList + Rank,
            SR: NameList + Rank,
        {
            type Output = NamedTensor<B, SL, DL>;

            fn $method(self, rhs: NamedTensor<B, SR, DR>) -> Self::Output {
                let r = align_to(rhs.inner, &rhs.names, &self.names);
                NamedTensor::new(self.inner $op r)
            }
        }
    };
}

impl_op!(Add, add, +);
impl_op!(Sub, sub, -);
impl_op!(Mul, mul, *);
impl_op!(Div, div, /);

/// `DNil` → `f32`, any non-empty dim list → `NamedTensor<B, Self, D>`.
pub trait NamedOut<B: Backend, const D: usize>: Sized {
    type Out;
    fn assemble(flat: Tensor<B, 1>, shape: [usize; D]) -> Self::Out;
}

impl<B: Backend> NamedOut<B, 0> for DNil
where
    B::FloatElem: Into<f32>,
{
    type Out = f32;
    fn assemble(flat: Tensor<B, 1>, _: [usize; 0]) -> f32 {
        flat.into_scalar().into()
    }
}

impl<B: Backend, H: DimName, T, const D: usize> NamedOut<B, D> for DCons<H, T>
where
    DCons<H, T>: NameList + Rank,
{
    type Out = NamedTensor<B, Self, D>;
    fn assemble(flat: Tensor<B, 1>, shape: [usize; D]) -> Self::Out {
        NamedTensor::new(flat.reshape(shape))
    }
}

pub type Named<B, S, const D: usize> = <S as NamedOut<B, D>>::Out;

/// Return type for [`dot`]. Implemented for `f32` (scalar contraction, all
/// dims shared) and `NamedTensor<B, S, D>` (partial contraction).
///
/// The associated type `Dims` is the output's dimension list; [`dot`] uses
/// it inside the [`Exclusive`] bound so the compiler can disambiguate
/// shared vs. exclusive dims *before* resolving the full return type.
pub trait DotResult<B: Backend>: Sized {
    type Dims: NameList + Rank;
    fn assemble_dot(flat: Tensor<B, 1>, raw_shape: &[usize], raw_names: &[&'static str]) -> Self;
}

impl<B: Backend> DotResult<B> for f32
where
    B::FloatElem: Into<f32>,
{
    type Dims = DNil;
    fn assemble_dot(flat: Tensor<B, 1>, _raw_shape: &[usize], _raw_names: &[&'static str]) -> f32 {
        flat.into_scalar().into()
    }
}

impl<B: Backend, S: NameList + Rank, const D: usize> DotResult<B> for NamedTensor<B, S, D> {
    type Dims = S;
    fn assemble_dot(flat: Tensor<B, 1>, raw_shape: &[usize], raw_names: &[&'static str]) -> Self {
        let shape: [usize; D] = std::array::from_fn(|i| raw_shape[i]);
        let tensor: Tensor<B, D> = flat.reshape(shape);
        let out_names = S::names();
        let perm = build_perm(raw_names, &out_names);
        NamedTensor::new(permute_if_needed(tensor, &perm))
    }
}

macro_rules! def_binop {
    ($name:ident, $verb:expr, $op:tt) => {
        #[doc = concat!("Element-wise ", $verb, " with union broadcasting. Inputs may differ in rank.")]
        pub fn $name<B, Out, SL, SR, UIdx, const DL: usize, const DR: usize, const D_OUT: usize>(
            lhs: NamedTensor<B, SL, DL>,
            rhs: NamedTensor<B, SR, DR>,
        ) -> NamedTensor<B, Out, D_OUT>
        where
            B: Backend,
            Out: IsUnionOf<SL, SR, UIdx> + NameList + Rank,
            SL: NameList + Rank,
            SR: NameList + Rank,
        {
            let out_names = Out::names();
            let l = align_to(lhs.inner, &lhs.names, &out_names);
            let r = align_to(rhs.inner, &rhs.names, &out_names);
            NamedTensor::new(l $op r)
        }
    };
}

def_binop!(add, "add", +);
def_binop!(sub, "subtract", -);
def_binop!(mul, "multiply", *);
def_binop!(div, "divide", /);

/// Tensor contraction over a single named dim `K`. Ranks of lhs, rhs, and output may differ.
pub fn matmul<
    B,
    K,
    Out,
    SL,
    SR,
    KIdxL,
    KIdxR,
    UIdx,
    const DL: usize,
    const DR: usize,
    const D_OUT: usize,
>(
    lhs: NamedTensor<B, SL, DL>,
    rhs: NamedTensor<B, SR, DR>,
    _contract: K,
) -> NamedTensor<B, Out, D_OUT>
where
    B: Backend,
    K: DimName,
    SL: Contains<K, KIdxL> + Remove<K, KIdxL> + NameList + Rank,
    SR: Contains<K, KIdxR> + Remove<K, KIdxR> + NameList + Rank,
    Out: IsUnionOf<<SL as Remove<K, KIdxL>>::Output, <SR as Remove<K, KIdxR>>::Output, UIdx>
        + NameList
        + Rank,
{
    let lhs_names = SL::names();
    let rhs_names = SR::names();
    let out_names = Out::names();
    let k = K::NAME;

    let k_size = lhs.inner.shape().dims[find_axis(&lhs_names, k)];
    assert_eq!(
        k_size,
        rhs.inner.shape().dims[find_axis(&rhs_names, k)],
        "matmul: K='{k}' size mismatch",
    );

    let batch: Vec<&'static str> = lhs_names
        .iter()
        .copied()
        .filter(|&n| n != k && rhs_names.contains(&n))
        .collect();
    let m: Vec<&'static str> = lhs_names
        .iter()
        .copied()
        .filter(|&n| n != k && !rhs_names.contains(&n))
        .collect();
    let n: Vec<&'static str> = rhs_names
        .iter()
        .copied()
        .filter(|&n| n != k && !lhs_names.contains(&n))
        .collect();

    let lhs_target: Vec<&'static str> = batch
        .iter()
        .chain(&m)
        .copied()
        .chain(std::iter::once(k))
        .collect();
    let rhs_target: Vec<&'static str> = batch
        .iter()
        .copied()
        .chain(std::iter::once(k))
        .chain(n.iter().copied())
        .collect();

    let lhs_p = permute_if_needed(lhs.inner, &build_perm(&lhs_names, &lhs_target));
    let rhs_p = permute_if_needed(rhs.inner, &build_perm(&rhs_names, &rhs_target));

    let lhs_shape = lhs_p.shape().dims;
    let rhs_shape = rhs_p.shape().dims;
    let batch_sizes: Vec<usize> = lhs_shape[..batch.len()].to_vec();
    let m_sizes: Vec<usize> = lhs_shape[batch.len()..batch.len() + m.len()].to_vec();
    let n_sizes: Vec<usize> = rhs_shape[batch.len() + 1..].to_vec();

    let prod = |xs: &[usize]| xs.iter().product::<usize>();
    let (batch_prod, m_prod, n_prod) = (prod(&batch_sizes), prod(&m_sizes), prod(&n_sizes));

    let lhs3: Tensor<B, 3> = lhs_p.reshape([batch_prod, m_prod, k_size]);
    let rhs3: Tensor<B, 3> = rhs_p.reshape([batch_prod, k_size, n_prod]);
    let raw3: Tensor<B, 3> = lhs3.matmul(rhs3);

    let raw_names: Vec<&'static str> = batch.iter().chain(&m).chain(&n).copied().collect();
    let raw_shape: [usize; D_OUT] = std::array::from_fn(|i| {
        if i < batch.len() {
            batch_sizes[i]
        } else if i < batch.len() + m.len() {
            m_sizes[i - batch.len()]
        } else {
            n_sizes[i - batch.len() - m.len()]
        }
    });
    let raw_out: Tensor<B, D_OUT> = raw3.reshape(raw_shape);

    NamedTensor::new(permute_if_needed(raw_out, &build_perm(&raw_names, &out_names)))
}

/// Contraction over every shared dim between `lhs` and `rhs`.
///
/// Each side may carry dims the other does not. Shared dims are contracted
/// (summed out); exclusive dims from both sides are kept in the output.
///
/// At compile time, [`Exclusive`] checks that every dim in each input is either
/// **shared** (present in the other input, contracted away) or **exclusive**
/// (present in the output dims, kept). If a dim appears in both the other input
/// AND the output, the compiler reports ambiguity — a contracted dim shouldn't
/// survive. If a dim is in neither, it reports an unsatisfied bound.
///
/// The return type drives inference: annotate as `f32` for a full contraction
/// (all dims shared) or as `NamedTensor<B, dims![…], D>` for a partial one.
///
/// ```text
/// lhs: dims![Batch, Features]
/// rhs: dims![Features, Classes]
///   → shared = Features (contracted)
///   → output = dims![Batch, Classes]
/// ```
pub fn dot<B, SL, SR, Ret, LIdx, RIdx, const DL: usize, const DR: usize>(
    lhs: NamedTensor<B, SL, DL>,
    rhs: NamedTensor<B, SR, DR>,
) -> Ret
where
    B: Backend,
    Ret: DotResult<B>,
    SL: NameList + Rank + Exclusive<SR, Ret::Dims, LIdx>,
    SR: NameList + Rank + Exclusive<SL, Ret::Dims, RIdx>,
{
    let lhs_names = SL::names();
    let rhs_names = SR::names();

    let shared: Vec<&'static str> = lhs_names
        .iter()
        .copied()
        .filter(|n| rhs_names.contains(n))
        .collect();
    let m: Vec<&'static str> = lhs_names
        .iter()
        .copied()
        .filter(|n| !rhs_names.contains(n))
        .collect();
    let n: Vec<&'static str> = rhs_names
        .iter()
        .copied()
        .filter(|n| !lhs_names.contains(n))
        .collect();

    // Runtime size check for shared dims
    for &k in &shared {
        let l_size = lhs.inner.shape().dims[find_axis(&lhs_names, k)];
        let r_size = rhs.inner.shape().dims[find_axis(&rhs_names, k)];
        assert_eq!(l_size, r_size, "dot: shared dim '{k}' size mismatch");
    }

    // Permute lhs to [m..., shared...] and rhs to [shared..., n...]
    let lhs_target: Vec<&'static str> = m.iter().chain(&shared).copied().collect();
    let rhs_target: Vec<&'static str> = shared.iter().chain(&n).copied().collect();

    let lhs_p = permute_if_needed(lhs.inner, &build_perm(&lhs_names, &lhs_target));
    let rhs_p = permute_if_needed(rhs.inner, &build_perm(&rhs_names, &rhs_target));

    let lhs_shape = lhs_p.shape().dims;
    let rhs_shape = rhs_p.shape().dims;

    let m_sizes: Vec<usize> = lhs_shape[..m.len()].to_vec();
    let n_sizes: Vec<usize> = rhs_shape[shared.len()..].to_vec();
    let shared_prod: usize = shared
        .iter()
        .enumerate()
        .map(|(i, _)| lhs_shape[m.len() + i])
        .product::<usize>()
        .max(1);

    let m_prod: usize = m_sizes.iter().product::<usize>().max(1);
    let n_prod: usize = n_sizes.iter().product::<usize>().max(1);

    // Contract via batched matmul: [m_prod, shared_prod] × [shared_prod, n_prod]
    let lhs2: Tensor<B, 2> = lhs_p.reshape([m_prod, shared_prod]);
    let rhs2: Tensor<B, 2> = rhs_p.reshape([shared_prod, n_prod]);
    let result2: Tensor<B, 2> = lhs2.matmul(rhs2);

    let total = m_prod * n_prod;
    let flat: Tensor<B, 1> = result2.reshape([total]);

    let raw_shape: Vec<usize> = m_sizes.iter().chain(&n_sizes).copied().collect();
    let raw_names: Vec<&'static str> = m.iter().chain(&n).copied().collect();

    Ret::assemble_dot(flat, &raw_shape, &raw_names)
}

/// Sum-reduce over named dim `C`.
pub fn sum<B, C, Out, S, Idx, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, S, D>,
) -> <Out as NamedOut<B, D_OUT>>::Out
where
    B: Backend,
    C: DimName,
    S: NameList + Rank + Contains<C, Idx> + Remove<C, Idx, Output = Out>,
    Out: NamedOut<B, D_OUT>,
{
    let axis = find_axis(&S::names(), C::NAME);
    let reduced: Tensor<B, D> = t.inner.sum_dim(axis);

    let red_shape = reduced.shape().dims;
    let out_prod: usize = red_shape.iter().product();
    let out_shape: [usize; D_OUT] = std::array::from_fn(|i| {
        if i < axis {
            red_shape[i]
        } else {
            red_shape[i + 1]
        }
    });

    let flat: Tensor<B, 1> = reduced.reshape([out_prod]);
    <Out as NamedOut<B, D_OUT>>::assemble(flat, out_shape)
}

/// Mean-reduce over named dims `Ks`.
pub fn mean<B, Ks, Out, S, Idx, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, S, D>,
) -> <Out as NamedOut<B, D_OUT>>::Out
where
    B: Backend,
    Ks: NameList,
    S: NameList + Rank + RemoveAll<Ks, Idx, Output = Out>,
    Out: NamedOut<B, D_OUT>,
{
    let s_names = S::names();
    let k_names = Ks::names();
    let mut inner = t.inner;
    for k in &k_names {
        let axis = find_axis(&s_names, k);
        inner = inner.mean_dim(axis);
    }
    let shape = inner.shape().dims;
    let kept: Vec<usize> = s_names
        .iter()
        .enumerate()
        .filter(|(_, n)| !k_names.contains(n))
        .map(|(i, _)| shape[i])
        .collect();
    let out_shape: [usize; D_OUT] = std::array::from_fn(|i| kept[i]);
    let prod: usize = out_shape.iter().product::<usize>().max(1);
    let flat: Tensor<B, 1> = inner.reshape([prod]);
    <Out as NamedOut<B, D_OUT>>::assemble(flat, out_shape)
}

/// Permute dims to a new order. `Out` must be a permutation of `S`.
pub fn permute<B, Out, S, FIdx, BIdx, const D: usize>(
    t: NamedTensor<B, S, D>,
) -> NamedTensor<B, Out, D>
where
    B: Backend,
    S: Subset<Out, FIdx> + NameList + Rank,
    Out: Subset<S, BIdx> + NameList + Rank,
{
    let from = S::names();
    let to = Out::names();
    let perm = build_perm(&from, &to);
    NamedTensor::new(permute_if_needed(t.inner, &perm))
}

/// Rename dim `Old` to `New` — zero cost.
pub fn rename<B, Old, New, Out, S, Idx, const D: usize>(
    t: NamedTensor<B, S, D>,
) -> NamedTensor<B, Out, D>
where
    B: Backend,
    S: Contains<Old, Idx> + ReplaceFirst<Old, New, Idx, Output = Out>,
    Out: NameList + Rank,
{
    NamedTensor::new(t.inner)
}
