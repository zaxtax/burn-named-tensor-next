//! # named-tensor
//!
//! Named-dimension tensors backed by [`burn`], following the Harvard NLP
//! named-tensor algebra on **stable Rust**.
//!
//! ## Harvard NLP rules
//!
//! | Operation              | Output dims                                           |
//! |------------------------|-------------------------------------------------------|
//! | `add(A, B)`            | **union(SA, SB)** — dims absent from one side get    |
//! |                        | size-1 axes inserted for broadcasting; ranks may differ|
//! | `matmul::<K>(A, B)`    | **union(SA, SB) − {K}** — K may be anywhere in either |
//! |                        | tensor; found at runtime, permuted, then contracted   |
//! | `dot(u, v)`            | scalar — rank-1 contraction                           |
//! | `sum_dim::<C>(T, i)`   | **S − {C}**                                           |
//! | `rename::<Old,New>(T)` | same shape, one marker swapped                        |
//!
//! ## Compile-time name safety
//!
//! Dim names are **zero-sized marker types** that also implement [`DimName`]
//! to expose a `&'static str` at runtime.  This bridges compile-time shape
//! algebra (overlap-free via the Index Trick) and runtime axis arithmetic
//! (unsqueeze / permute).
//!
//! ## The Index Trick (frunk pattern)
//!
//! `Contains`, `Remove`, `Union`, … use a phantom `Index` type parameter to
//! make overlapping impls disjoint: `Here` selects the base case, `There<I>`
//! the recursive case.  The compiler infers `Index`; callers never write it.

use burn::prelude::*;
use std::marker::PhantomData;

// ═══════════════════════════════════════════════════════════════════════════
// §1  DimName — runtime name string attached to each dim marker
// ═══════════════════════════════════════════════════════════════════════════

/// Every dim marker must expose its name as a `&'static str`.
/// Use the [`dim!`] macro to declare markers without boilerplate.
pub trait DimName {
    const NAME: &'static str;
}

/// Declare dim marker types and automatically impl `DimName` for each.
///
/// ```ignore
/// dim!(Batch, M, K, N);
/// ```
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

// ═══════════════════════════════════════════════════════════════════════════
// §2  Index types for the frunk overlap-resolution trick
// ═══════════════════════════════════════════════════════════════════════════

pub struct Here;
pub struct There<I>(PhantomData<I>);

// ═══════════════════════════════════════════════════════════════════════════
// §3  Type-level dimension list
// ═══════════════════════════════════════════════════════════════════════════

pub struct DNil;
pub struct DCons<H, T>(PhantomData<fn() -> (H, T)>);

/// Build a dim list from marker types.
///
/// ```ignore
/// type S = dims![Batch, M, K];
/// //      = DCons<Batch, DCons<M, DCons<K, DNil>>>
/// ```
#[macro_export]
macro_rules! dims {
    ()                  => { $crate::DNil };
    ($h:ty)             => { $crate::DCons<$h, $crate::DNil> };
    ($h:ty, $($t:ty),+) => { $crate::DCons<$h, $crate::dims![$($t),+]> };
}

// ── Rank ───────────────────────────────────────────────────────────────────

pub trait Rank { const RANK: usize; }
impl Rank for DNil { const RANK: usize = 0; }
impl<H, T: Rank> Rank for DCons<H, T> { const RANK: usize = 1 + T::RANK; }


// ── NameList — produce a runtime Vec of dim name strings ──────────────────

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

// ── Contains<D, Idx> ──────────────────────────────────────────────────────

pub trait Contains<D, Idx> {}
impl<D, T>       Contains<D, Here>      for DCons<D, T> {}
impl<H, D, T, I> Contains<D, There<I>> for DCons<H, T>
where T: Contains<D, I> {}

// ── Remove<D, Idx> ────────────────────────────────────────────────────────

pub trait Remove<D, Idx> { type Output; }
impl<D, T> Remove<D, Here> for DCons<D, T> {
    type Output = T;
}
impl<H, D, T, I> Remove<D, There<I>> for DCons<H, T>
where T: Remove<D, I> {
    type Output = DCons<H, <T as Remove<D, I>>::Output>;
}

// ── Append<D> ─────────────────────────────────────────────────────────────

pub trait Append<D> { type Output; }
impl<D> Append<D> for DNil { type Output = DCons<D, DNil>; }
impl<H, T: Append<D>, D> Append<D> for DCons<H, T> {
    type Output = DCons<H, <T as Append<D>>::Output>;
}

// ── Subset<Out, Idx> ──────────────────────────────────────────────────────
//
// Verify that every element of Self appears in Out (using Contains).
// Contains resolves unambiguously for lists without duplicates, so this
// trait never triggers E0283 ambiguity.

pub trait Subset<Out, Idx> {}
impl<Out> Subset<Out, ()> for DNil {}
impl<H, T, Out, IC, IT> Subset<Out, (IC, IT)> for DCons<H, T>
where
    Out: Contains<H, IC>,
    T: Subset<Out, IT>,
{}

// ── IsUnionOf<SL, SR, Idx> ───────────────────────────────────────────────
//
// Verification-based union: instead of computing L ∪ R, the caller
// provides the output type Out and we verify Out ⊇ SL and Out ⊇ SR.
//
// This uses only Contains (which is always deterministic for duplicate-free
// dim lists), avoiding the solver ambiguity that arises with "found/not
// found" dispatch.

pub trait IsUnionOf<SL, SR, Idx> {}
impl<Out, SL, SR, LIdx, RIdx> IsUnionOf<SL, SR, (LIdx, RIdx)> for Out
where
    SL: Subset<Out, LIdx>,
    SR: Subset<Out, RIdx>,
{}

// ── ReplaceFirst<Old, New, Idx> ───────────────────────────────────────────

pub trait ReplaceFirst<Old, New, Idx> { type Output; }
impl<Old, New, T> ReplaceFirst<Old, New, Here> for DCons<Old, T> {
    type Output = DCons<New, T>;
}
impl<H, Old, New, T, I> ReplaceFirst<Old, New, There<I>> for DCons<H, T>
where T: ReplaceFirst<Old, New, I> {
    type Output = DCons<H, <T as ReplaceFirst<Old, New, I>>::Output>;
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  Runtime axis utilities
// ═══════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════
// §5  NamedTensor
// ═══════════════════════════════════════════════════════════════════════════

/// A burn `Tensor<B, D>` labelled with compile-time named dim list `S`.
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

// ═══════════════════════════════════════════════════════════════════════════
// §6  Operations
// ═══════════════════════════════════════════════════════════════════════════

// ── §6.1  add — union broadcast, different ranks OK ───────────────────────
//
// Algorithm:
//   1. out_names = Union(SL, SR)::names()  [consistent with type-level Union]
//   2. For each operand:
//        a. unsqueeze_dims at positions in out_names absent from this operand.
//        b. Permute to match out_names order.
//   3. burn's `+` broadcasts over all size-1 axes.

/// Element-wise add.  **Output shape = union(SL, SR).**
///
/// Dims absent from one operand get size-1 axes inserted automatically.
/// The two input tensors may have different ranks (`DL` ≠ `DR`).
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

    // Expand one operand's tensor to rank D_OUT, inserting size-1 axes for
    // dims in out_names that are absent from operand_names, then permute.
    fn align<B: Backend, const D_IN: usize, const D_OUT: usize>(
        t: Tensor<B, D_IN>,
        operand_names: &[&'static str],
        out_names: &[&'static str],
    ) -> Tensor<B, D_OUT> {
        // Positions in the output where this operand has no dim → insert size-1.
        let missing: Vec<isize> = out_names
            .iter()
            .enumerate()
            .filter(|(_, n)| !operand_names.contains(n))
            .map(|(i, _)| i as isize)
            .collect();

        let expanded: Tensor<B, D_OUT> = t.unsqueeze_dims(&missing);

        // Reconstruct what name order looks like after the insertions.
        let mut current = vec![""; D_OUT];
        let mut src = 0usize;
        for i in 0..D_OUT {
            if missing.contains(&(i as isize)) {
                current[i] = out_names[i]; // size-1 placeholder for this output dim
            } else {
                current[i] = operand_names[src];
                src += 1;
            }
        }

        // Permute so axis order matches out_names.
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

// ── §6.2  matmul — contract K from any axis position ─────────────────────
//
// Both tensors must have the same burn rank D.
//
// Algorithm:
//   1. Find K's axis in lhs and rhs at runtime via DimName.
//   2. Permute lhs → (...non-K in order..., K)
//      Permute rhs → (...non-K except last..., K, last-non-K)
//      so burn sees (..., M, K) × (..., K, N).
//   3. Call burn's matmul (which broadcasts batch dims automatically).
//   4. Track the raw output dim-name order; permute to match the
//      type-level `union(SL,SR) − {K}` order.
//
// Compile-time guarantees:
//   K ∈ SL  →  SL: Contains<K, _>   (error if K absent from lhs)
//   K ∈ SR  →  SR: Contains<K, _>   (error if K absent from rhs)
//   Output shape = Union(SL,SR) − {K}  (verified by type inference)

/// Matrix multiply contracting over dim `K`.
///
/// `K` may be at **any axis position** in either tensor.
/// **Output shape = union(SL, SR) − {K}.**
/// Both tensors must have the same burn rank `D`.
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

    // Runtime size check.
    {
        let ls = lhs.inner.shape();
        let rs = rhs.inner.shape();
        assert_eq!(
            ls.dims[k_l], rs.dims[k_r],
            "matmul: K='{k}' size mismatch: lhs[{k_l}]={}, rhs[{k_r}]={}",
            ls.dims[k_l], rs.dims[k_r],
        );
    }

    // Classify dims into batch (shared, ≠K), M-only (lhs only), N-only (rhs only).
    let rhs_set: std::collections::HashSet<&'static str> = rhs_names.iter().copied().collect();
    let lhs_set: std::collections::HashSet<&'static str> = lhs_names.iter().copied().collect();

    let batch_names: Vec<&'static str> = lhs_names.iter()
        .filter(|&&n| n != k && rhs_set.contains(n))
        .copied().collect();
    let m_names: Vec<&'static str> = lhs_names.iter()
        .filter(|&&n| n != k && !rhs_set.contains(n))
        .copied().collect();
    let n_names: Vec<&'static str> = rhs_names.iter()
        .filter(|&&n| n != k && !lhs_set.contains(n))
        .copied().collect();

    // Permute lhs to [batch, M, K] and rhs to [batch, K, N] so burn's matmul
    // sees aligned batch dims and contracts the last two axes correctly.
    let lhs_target: Vec<&'static str> = batch_names.iter()
        .chain(m_names.iter()).chain(std::iter::once(&k))
        .copied().collect();
    let rhs_target: Vec<&'static str> = batch_names.iter()
        .chain(std::iter::once(&k)).chain(n_names.iter())
        .copied().collect();

    let lhs_perm = build_perm(&lhs_names, &lhs_target);
    let rhs_perm = build_perm(&rhs_names, &rhs_target);

    let lhs_p = if is_identity(&lhs_perm) { lhs.inner } else {
        let arr: [isize; D] = lhs_perm.iter().map(|&x| x as isize).collect::<Vec<_>>().try_into().unwrap();
        lhs.inner.permute(arr)
    };
    let rhs_p = if is_identity(&rhs_perm) { rhs.inner } else {
        let arr: [isize; D] = rhs_perm.iter().map(|&x| x as isize).collect::<Vec<_>>().try_into().unwrap();
        rhs.inner.permute(arr)
    };

    // burn matmul output: [batch, M, N]
    let raw_names: Vec<&'static str> = batch_names.iter()
        .chain(m_names.iter()).chain(n_names.iter())
        .copied().collect();

    let raw: Tensor<B, D> = lhs_p.matmul(rhs_p);

    // Permute raw to match out_names (the type-level union − K order).
    let perm = build_perm(&raw_names, &out_names);
    let raw_out: Tensor<B, D> = if is_identity(&perm) {
        raw
    } else {
        let perm_arr: [isize; D] = perm.iter().map(|&x| x as isize).collect::<Vec<_>>().try_into().unwrap();
        raw.permute(perm_arr)
    };
    // D == D_OUT in all well-formed cases (burn's matmul preserves rank).
    // We assert at runtime; proving D == D_OUT at compile time requires
    // `generic_const_exprs` (nightly).
    assert_eq!(D, D_OUT, "matmul: D={D} ≠ D_OUT={D_OUT} (internal error)");
    // SAFETY: We just asserted D == D_OUT.  Tensor<B,D> is layout-identical
    // to Tensor<B,D_OUT> when D == D_OUT because D is phantom in burn.
    let result: Tensor<B, D_OUT> = unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(raw_out)) };

    NamedTensor::new(result)
}

// ── §6.3  dot ─────────────────────────────────────────────────────────────

/// Dot product of two rank-1 tensors sharing the same dim marker.
pub fn dot<B, S>(
    lhs: NamedTensor<B, S, 1>,
    rhs: NamedTensor<B, S, 1>,
) -> f32
where
    B: Backend,
    S: Rank,
    B::FloatElem: Into<f32>,
{
    let ls = lhs.inner.shape().dims[0];
    let rs = rhs.inner.shape().dims[0];
    assert_eq!(ls, rs, "dot: size mismatch ({ls} vs {rs})");
    (lhs.inner * rhs.inner).sum().into_scalar().into()
}

// ── §6.4  sum_dim ─────────────────────────────────────────────────────────

/// Sum-reduce over named dim `C`.  `dim_index` is the burn axis (0-based).
pub fn sum_dim<B, C, Out, S, Idx, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, S, D>,
    dim_index: usize,
) -> NamedTensor<B, Out, D_OUT>
where
    B:   Backend,
    S:   Contains<C, Idx> + Remove<C, Idx, Output = Out>,
    Out: Rank,
{
    let summed:   Tensor<B, D>     = t.inner.sum_dim(dim_index);
    let squeezed: Tensor<B, D_OUT> = summed.squeeze(dim_index);
    NamedTensor::new(squeezed)
}

// ── §6.5  rename ──────────────────────────────────────────────────────────

/// Rename dim `Old` to `New` — zero runtime cost, purely type-level.
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
