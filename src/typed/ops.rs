use burn::prelude::*;

use super::dims::*;
use super::tensor::NamedTensor;

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

/// Inverse of [`NamedOut`]: given a concrete result type (`f32` or
/// `NamedTensor<B, S, D>`), recovers the dimension list and knows how to
/// assemble the value from a flat tensor.
///
/// This lets generic functions use the *return-type annotation* to drive
/// trait-solver inference — the solver resolves `Ret` first (it's the
/// return type), then reads `Ret::Dims` to feed into downstream bounds
/// like [`Exclusive`].
pub trait IntoNamedResult<B: Backend>: Sized {
    type Dims: NameList + Rank;
    fn assemble(flat: Tensor<B, 1>, raw_shape: &[usize], raw_names: &[&'static str]) -> Self;
}

impl<B: Backend> IntoNamedResult<B> for f32
where
    B::FloatElem: Into<f32>,
{
    type Dims = DNil;
    fn assemble(flat: Tensor<B, 1>, _raw_shape: &[usize], _raw_names: &[&'static str]) -> f32 {
        flat.into_scalar().into()
    }
}

impl<B: Backend, S: NameList + Rank, const D: usize> IntoNamedResult<B> for NamedTensor<B, S, D> {
    type Dims = S;
    fn assemble(flat: Tensor<B, 1>, raw_shape: &[usize], raw_names: &[&'static str]) -> Self {
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

/// Tensor contraction over shared dims not present in the output.
///
/// Shared dims that **are** in the output become batch dims; shared dims
/// that are **not** in the output are contracted. The return type drives
/// inference — annotate as `NamedTensor<B, dims![…], D>` for a partial
/// contraction or `f32` for a full one.
///
/// ```text
/// lhs: dims![Batch, M, K]
/// rhs: dims![Batch, K, N]
///   → contracted = K (in both inputs, not in output)
///   → batch      = Batch (in both inputs and output)
///   → output     = dims![Batch, M, N]
/// ```
pub fn matmul<B, SL, SR, Ret, const DL: usize, const DR: usize>(
    lhs: NamedTensor<B, SL, DL>,
    rhs: NamedTensor<B, SR, DR>,
) -> Ret
where
    B: Backend,
    Ret: IntoNamedResult<B>,
    Ret::Dims: NameList + Rank,
    SL: NameList + Rank,
    SR: NameList + Rank,
{
    let lhs_names = SL::names();
    let rhs_names = SR::names();
    let out_names = Ret::Dims::names();

    let contracted: Vec<&'static str> = lhs_names
        .iter()
        .copied()
        .filter(|n| rhs_names.contains(n) && !out_names.contains(n))
        .collect();
    let batch: Vec<&'static str> = lhs_names
        .iter()
        .copied()
        .filter(|n| rhs_names.contains(n) && out_names.contains(n))
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

    assert!(
        !contracted.is_empty(),
        "matmul: no dims to contract between {:?} and {:?} given output {:?}",
        lhs_names,
        rhs_names,
        out_names,
    );

    for &d in &lhs_names {
        assert!(
            rhs_names.contains(&d) || out_names.contains(&d),
            "matmul: lhs dim '{d}' is not in rhs or output",
        );
    }
    for &d in &rhs_names {
        assert!(
            lhs_names.contains(&d) || out_names.contains(&d),
            "matmul: rhs dim '{d}' is not in lhs or output",
        );
    }
    for &d in &out_names {
        assert!(
            lhs_names.contains(&d) || rhs_names.contains(&d),
            "matmul: output dim '{d}' is not in either input",
        );
    }

    for &k in &contracted {
        let l_size = lhs.inner.shape().to_vec()[find_axis(&lhs_names, k)];
        let r_size = rhs.inner.shape().to_vec()[find_axis(&rhs_names, k)];
        assert_eq!(l_size, r_size, "matmul: contracted dim '{k}' size mismatch");
    }

    // lhs → [batch..., m..., contracted...]
    let lhs_target: Vec<&'static str> =
        batch.iter().chain(&m).chain(&contracted).copied().collect();
    // rhs → [batch..., contracted..., n...]
    let rhs_target: Vec<&'static str> =
        batch.iter().chain(&contracted).chain(&n).copied().collect();

    let lhs_p = permute_if_needed(lhs.inner, &build_perm(&lhs_names, &lhs_target));
    let rhs_p = permute_if_needed(rhs.inner, &build_perm(&rhs_names, &rhs_target));

    let lhs_shape = lhs_p.shape().to_vec();
    let rhs_shape = rhs_p.shape().to_vec();
    let batch_sizes: Vec<usize> = lhs_shape[..batch.len()].to_vec();
    let m_sizes: Vec<usize> = lhs_shape[batch.len()..batch.len() + m.len()].to_vec();
    let n_sizes: Vec<usize> = rhs_shape[batch.len() + contracted.len()..].to_vec();

    let prod = |xs: &[usize]| xs.iter().product::<usize>().max(1);
    let (batch_prod, m_prod, n_prod) = (prod(&batch_sizes), prod(&m_sizes), prod(&n_sizes));
    let k_prod: usize = lhs_shape[batch.len() + m.len()..]
        .iter()
        .product::<usize>()
        .max(1);

    let lhs3: Tensor<B, 3> = lhs_p.reshape([batch_prod, m_prod, k_prod]);
    let rhs3: Tensor<B, 3> = rhs_p.reshape([batch_prod, k_prod, n_prod]);
    let raw3: Tensor<B, 3> = lhs3.matmul(rhs3);

    let raw_names: Vec<&'static str> = batch.iter().chain(&m).chain(&n).copied().collect();
    let raw_shape: Vec<usize> = batch_sizes
        .iter()
        .chain(&m_sizes)
        .chain(&n_sizes)
        .copied()
        .collect();

    let total: usize = raw_shape.iter().product::<usize>().max(1);
    let flat: Tensor<B, 1> = raw3.reshape([total]);

    Ret::assemble(flat, &raw_shape, &raw_names)
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
    Ret: IntoNamedResult<B>,
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
        let l_size = lhs.inner.shape().to_vec()[find_axis(&lhs_names, k)];
        let r_size = rhs.inner.shape().to_vec()[find_axis(&rhs_names, k)];
        assert_eq!(l_size, r_size, "dot: shared dim '{k}' size mismatch");
    }

    // Permute lhs to [m..., shared...] and rhs to [shared..., n...]
    let lhs_target: Vec<&'static str> = m.iter().chain(&shared).copied().collect();
    let rhs_target: Vec<&'static str> = shared.iter().chain(&n).copied().collect();

    let lhs_p = permute_if_needed(lhs.inner, &build_perm(&lhs_names, &lhs_target));
    let rhs_p = permute_if_needed(rhs.inner, &build_perm(&rhs_names, &rhs_target));

    let lhs_shape = lhs_p.shape().to_vec();
    let rhs_shape = rhs_p.shape().to_vec();

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

    Ret::assemble(flat, &raw_shape, &raw_names)
}

fn reduce_impl<B, Ks, Out, S, Idx, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, S, D>,
    mut f: impl FnMut(Tensor<B, D>, usize) -> Tensor<B, D>,
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
        inner = f(inner, find_axis(&s_names, k));
    }
    let shape = inner.shape().to_vec();
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

macro_rules! def_reduce {
    ($name:ident, $dim_op:ident) => {
        #[doc = concat!(stringify!($dim_op), "-reduce over named dims `Ks`.")]
        pub fn $name<B, Ks, Out, S, Idx, const D: usize, const D_OUT: usize>(
            t: NamedTensor<B, S, D>,
        ) -> <Out as NamedOut<B, D_OUT>>::Out
        where
            B: Backend,
            Ks: NameList,
            S: NameList + Rank + RemoveAll<Ks, Idx, Output = Out>,
            Out: NamedOut<B, D_OUT>,
        {
            reduce_impl::<B, Ks, Out, S, Idx, D, D_OUT>(t, |inner, axis| inner.$dim_op(axis))
        }
    };
}

def_reduce!(sum, sum_dim);
def_reduce!(max, max_dim);
def_reduce!(min, min_dim);
def_reduce!(prod, prod_dim);
def_reduce!(mean, mean_dim);

/// Argmax over named dim `C`. Returns a plain `Tensor<B, D_OUT, Int>` of indices.
pub fn argmax<B, C, S, Idx, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, S, D>,
) -> Tensor<B, D_OUT, burn::tensor::Int>
where
    B: Backend,
    C: DimName,
    S: NameList + Rank + Contains<C, Idx>,
{
    let axis = find_axis(&S::names(), C::NAME);
    t.inner.argmax(axis).squeeze_dim(axis)
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

/// Concatenate tensors along the existing dim `Dm`. All inputs share the
/// same dim list `S` (and thus the same axis order), so no alignment is
/// needed — the dim to grow is named, eliminating the positional `dim=`
/// foot-gun. The output has the same dim list `S`; the concat dim's size is
/// the sum of the inputs' sizes along it.
///
/// Panics if `tensors` is empty, or if any non-concat dim has mismatched
/// sizes across inputs.
pub fn concat<B, S, Dm, I, const D: usize>(
    tensors: Vec<NamedTensor<B, S, D>>,
    _dim: Dm,
) -> NamedTensor<B, S, D>
where
    B: Backend,
    Dm: DimName,
    S: NameList + Rank + Contains<Dm, I>,
{
    let axis = find_axis(&S::names(), Dm::NAME);
    let inners: Vec<Tensor<B, D>> = tensors.into_iter().map(|t| t.inner).collect();
    NamedTensor::new(Tensor::cat(inners, axis))
}

/// Stack tensors along a *new* dim `New`, prepended to the front of the dim
/// list. All inputs share the same dim list `S`; the output is
/// `DCons<New, S>` — the new dim gets a semantic name, unlike positional
/// `stack` where it is an anonymous axis 0.
///
/// `D_OUT` must equal `D + 1` (the new dim adds one rank). To place the new
/// dim elsewhere, `permute` the result.
///
/// Panics if `tensors` is empty, or if the inputs' shapes differ.
pub fn stack<B, S, New, const D: usize, const D_OUT: usize>(
    tensors: Vec<NamedTensor<B, S, D>>,
    _dim: New,
) -> NamedTensor<B, DCons<New, S>, D_OUT>
where
    B: Backend,
    New: DimName,
    S: NameList + Rank,
{
    debug_assert_eq!(D_OUT, D + 1, "stack: D_OUT must equal D + 1");
    let inners: Vec<Tensor<B, D>> = tensors.into_iter().map(|t| t.inner).collect();
    NamedTensor::new(Tensor::stack::<D_OUT>(inners, 0))
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

/// Call a typed reduction with only the dims to reduce specified; all other
/// type parameters are inferred from the input and return type.
///
/// ```ignore
/// let x: NamedTensor<B, dims![C], 1> = reduce!(sum, t, [H, W]);
/// let x: NamedTensor<B, dims![B, C], 2> = reduce!(max, t, [H]);
/// ```
#[macro_export]
macro_rules! reduce {
    ($func:ident, $t:expr, [$($dim:ident),* $(,)?] $(,)?) => {
        $crate::typed::ops::$func::<_, $crate::dims![$($dim),*], _, _, _, _, _>($t)
    };
}
