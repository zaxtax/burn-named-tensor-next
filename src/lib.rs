//! Named-dimension tensors backed by [`burn`] on stable Rust.
//!
//! Dim names are zero-sized marker types implementing [`DimName`].
//! Type-level list operations use the frunk index trick to resolve overlapping impls.

use burn::prelude::*;
use std::marker::PhantomData;

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

// Index types for the frunk overlap-resolution trick
pub struct Here;
pub struct There<I>(PhantomData<I>);

// Type-level dimension list
pub struct DNil;
pub struct DCons<H, T>(PhantomData<fn() -> (H, T)>);

#[macro_export]
macro_rules! dims {
    ()                  => { $crate::DNil };
    ($h:ty)             => { $crate::DCons<$h, $crate::DNil> };
    ($h:ty, $($t:ty),+) => { $crate::DCons<$h, $crate::dims![$($t),+]> };
}

pub trait Rank { const RANK: usize; }
impl Rank for DNil { const RANK: usize = 0; }
impl<H, T: Rank> Rank for DCons<H, T> { const RANK: usize = 1 + T::RANK; }

pub trait NameList {
    fn names() -> Vec<&'static str>;
}
impl NameList for DNil {
    fn names() -> Vec<&'static str> { vec![] }
}
impl<H: DimName, T: NameList> NameList for DCons<H, T> {
    fn names() -> Vec<&'static str> {
        let mut v = vec![H::NAME];
        v.extend(T::names());
        v
    }
}

pub trait Contains<D, Idx> {}
impl<D, T>       Contains<D, Here>     for DCons<D, T> {}
impl<H, D, T, I> Contains<D, There<I>> for DCons<H, T>
where T: Contains<D, I> {}

pub trait Remove<D, Idx> { type Output; }
impl<D, T> Remove<D, Here> for DCons<D, T> {
    type Output = T;
}
impl<H, D, T, I> Remove<D, There<I>> for DCons<H, T>
where T: Remove<D, I> {
    type Output = DCons<H, <T as Remove<D, I>>::Output>;
}

pub trait Append<D> { type Output; }
impl<D> Append<D> for DNil { type Output = DCons<D, DNil>; }
impl<H, T: Append<D>, D> Append<D> for DCons<H, T> {
    type Output = DCons<H, <T as Append<D>>::Output>;
}

pub trait Subset<Out, Idx> {}
impl<Out> Subset<Out, ()> for DNil {}
impl<H, T, Out, IC, IT> Subset<Out, (IC, IT)> for DCons<H, T>
where
    Out: Contains<H, IC>,
    T: Subset<Out, IT>,
{}

pub trait IsUnionOf<SL, SR, Idx> {}
impl<Out, SL, SR, LIdx, RIdx> IsUnionOf<SL, SR, (LIdx, RIdx)> for Out
where
    SL: Subset<Out, LIdx>,
    SR: Subset<Out, RIdx>,
{}

pub trait ReplaceFirst<Old, New, Idx> { type Output; }
impl<Old, New, T> ReplaceFirst<Old, New, Here> for DCons<Old, T> {
    type Output = DCons<New, T>;
}
impl<H, Old, New, T, I> ReplaceFirst<Old, New, There<I>> for DCons<H, T>
where T: ReplaceFirst<Old, New, I> {
    type Output = DCons<H, <T as ReplaceFirst<Old, New, I>>::Output>;
}

// Runtime axis utilities

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
    if is_identity(perm) { return t; }
    let arr: [isize; D] = perm.iter().map(|&x| x as isize)
        .collect::<Vec<_>>().try_into().unwrap();
    t.permute(arr)
}

// NamedTensor

pub struct NamedTensor<B: Backend, S, const D: usize> {
    pub inner: Tensor<B, D>,
    _s: PhantomData<fn() -> S>,
}

impl<B: Backend, S: Rank, const D: usize> NamedTensor<B, S, D> {
    pub fn new(t: Tensor<B, D>) -> Self {
        debug_assert_eq!(D, S::RANK);
        Self { inner: t, _s: PhantomData }
    }
    pub fn into_inner(self) -> Tensor<B, D> { self.inner }
    pub fn shape(&self) -> burn::tensor::Shape { self.inner.shape() }
}

impl<B: Backend, S, const D: usize> Clone for NamedTensor<B, S, D>
where Tensor<B, D>: Clone {
    fn clone(&self) -> Self { Self { inner: self.inner.clone(), _s: PhantomData } }
}
impl<B: Backend, S, const D: usize> std::fmt::Debug for NamedTensor<B, S, D>
where Tensor<B, D>: std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.inner.fmt(f) }
}
impl<B: Backend, S, const D: usize> std::fmt::Display for NamedTensor<B, S, D>
where Tensor<B, D>: std::fmt::Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.inner.fmt(f) }
}

// Operations

/// Element-wise add with union broadcasting. Inputs may differ in rank.
pub fn add<B, Out, SL, SR, UIdx, const DL: usize, const DR: usize, const D_OUT: usize>(
    lhs: NamedTensor<B, SL, DL>,
    rhs: NamedTensor<B, SR, DR>,
) -> NamedTensor<B, Out, D_OUT>
where
    B:   Backend,
    Out: IsUnionOf<SL, SR, UIdx> + NameList + Rank,
    SL:  NameList + Rank,
    SR:  NameList + Rank,
{
    let lhs_names = SL::names();
    let rhs_names = SR::names();
    let out_names = Out::names();

    fn align<B: Backend, const D_IN: usize, const D_OUT: usize>(
        t: Tensor<B, D_IN>,
        operand_names: &[&'static str],
        out_names: &[&'static str],
    ) -> Tensor<B, D_OUT> {
        let missing: Vec<isize> = out_names.iter().enumerate()
            .filter(|(_, n)| !operand_names.contains(n))
            .map(|(i, _)| i as isize)
            .collect();

        let expanded: Tensor<B, D_OUT> = t.unsqueeze_dims(&missing);

        let mut current = vec![""; D_OUT];
        let mut src = 0usize;
        for i in 0..D_OUT {
            if missing.contains(&(i as isize)) {
                current[i] = out_names[i];
            } else {
                current[i] = operand_names[src];
                src += 1;
            }
        }

        let perm = build_perm(&current, out_names);
        if is_identity(&perm) { expanded } else {
            let perm_arr: [isize; D_OUT] = perm.iter().map(|&x| x as isize)
                .collect::<Vec<_>>().try_into().unwrap();
            expanded.permute(perm_arr)
        }
    }

    let l = align(lhs.inner, &lhs_names, &out_names);
    let r = align(rhs.inner, &rhs_names, &out_names);
    NamedTensor::new(l + r)
}

/// Matrix multiply contracting over dim `K`. K may be at any axis position.
/// Both tensors must share the same burn rank `D`.
pub fn matmul<B, K, Out, SL, SR, KIdxL, KIdxR, UIdx,
              const D: usize, const D_OUT: usize>(
    lhs: NamedTensor<B, SL, D>,
    rhs: NamedTensor<B, SR, D>,
    _contract: K,
) -> NamedTensor<B, Out, D_OUT>
where
    B:   Backend,
    K:   DimName,
    SL:  Contains<K, KIdxL> + Remove<K, KIdxL> + NameList + Rank,
    SR:  Contains<K, KIdxR> + Remove<K, KIdxR> + NameList + Rank,
    Out: IsUnionOf<
             <SL as Remove<K, KIdxL>>::Output,
             <SR as Remove<K, KIdxR>>::Output,
             UIdx,
         > + NameList + Rank,
{
    let lhs_names = SL::names();
    let rhs_names = SR::names();
    let out_names = Out::names();

    let k = K::NAME;
    let k_l = find_axis(&lhs_names, k);
    let k_r = find_axis(&rhs_names, k);

    assert_eq!(
        lhs.inner.shape().dims[k_l], rhs.inner.shape().dims[k_r],
        "matmul: K='{k}' size mismatch",
    );

    let rhs_set: std::collections::HashSet<&str> = rhs_names.iter().copied().collect();
    let lhs_set: std::collections::HashSet<&str> = lhs_names.iter().copied().collect();

    let batch_names: Vec<_> = lhs_names.iter().filter(|&&n| n != k && rhs_set.contains(n)).copied().collect();
    let m_names: Vec<_> = lhs_names.iter().filter(|&&n| n != k && !rhs_set.contains(n)).copied().collect();
    let n_names: Vec<_> = rhs_names.iter().filter(|&&n| n != k && !lhs_set.contains(n)).copied().collect();

    // Permute to [batch, M, K] × [batch, K, N]
    let lhs_target: Vec<_> = batch_names.iter().chain(m_names.iter()).chain(std::iter::once(&k)).copied().collect();
    let rhs_target: Vec<_> = batch_names.iter().chain(std::iter::once(&k)).chain(n_names.iter()).copied().collect();

    let lhs_p = permute_if_needed(lhs.inner, &build_perm(&lhs_names, &lhs_target));
    let rhs_p = permute_if_needed(rhs.inner, &build_perm(&rhs_names, &rhs_target));

    let raw: Tensor<B, D> = lhs_p.matmul(rhs_p);
    let raw_names: Vec<_> = batch_names.iter().chain(m_names.iter()).chain(n_names.iter()).copied().collect();
    let raw_out: Tensor<B, D> = permute_if_needed(raw, &build_perm(&raw_names, &out_names));

    assert_eq!(D, D_OUT, "matmul: D={D} ≠ D_OUT={D_OUT}");
    // SAFETY: D == D_OUT; Tensor<B,D> is layout-identical to Tensor<B,D_OUT>.
    let result: Tensor<B, D_OUT> = unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(raw_out)) };
    NamedTensor::new(result)
}

/// Dot product of two rank-1 tensors sharing the same dim marker.
pub fn dot<B, S>(lhs: NamedTensor<B, S, 1>, rhs: NamedTensor<B, S, 1>) -> f32
where
    B: Backend,
    S: Rank,
    B::FloatElem: Into<f32>,
{
    assert_eq!(lhs.inner.shape().dims[0], rhs.inner.shape().dims[0], "dot: size mismatch");
    (lhs.inner * rhs.inner).sum().into_scalar().into()
}

/// Sum-reduce over named dim `C`. `dim_index` is the burn axis (0-based).
pub fn sum_dim<B, C, Out, S, Idx, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, S, D>,
    dim_index: usize,
) -> NamedTensor<B, Out, D_OUT>
where
    B:   Backend,
    S:   Contains<C, Idx> + Remove<C, Idx, Output = Out>,
    Out: Rank,
{
    let squeezed: Tensor<B, D_OUT> = t.inner.sum_dim(dim_index).squeeze();
    NamedTensor::new(squeezed)
}

/// Permute (transpose) dims to a new order — like xarray's `.transpose()`.
/// `Out` must be a permutation of `S` (same dims, different order).
pub fn permute<B, Out, S, FIdx, BIdx, const D: usize>(
    t: NamedTensor<B, S, D>,
) -> NamedTensor<B, Out, D>
where
    B:   Backend,
    S:   Subset<Out, FIdx> + NameList + Rank,
    Out: Subset<S, BIdx> + NameList + Rank,
{
    let from = S::names();
    let to   = Out::names();
    let perm = build_perm(&from, &to);
    NamedTensor::new(permute_if_needed(t.inner, &perm))
}

/// Rename dim `Old` to `New` — zero cost, purely type-level.
pub fn rename<B, Old, New, Out, S, Idx, const D: usize>(
    t: NamedTensor<B, S, D>,
) -> NamedTensor<B, Out, D>
where
    B:   Backend,
    S:   Contains<Old, Idx> + ReplaceFirst<Old, New, Idx, Output = Out>,
    Out: Rank,
{
    NamedTensor::new(t.inner)
}
