use burn::prelude::*;
use std::collections::HashSet;

use super::tensor::NamedTensor;

pub(crate) fn axis_of(names: &[String], name: &str) -> usize {
    names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("named-tensor: dim '{name}' not found in {names:?}"))
}

pub(crate) fn perm_of(from: &[String], to: &[String]) -> Vec<usize> {
    to.iter().map(|n| axis_of(from, n)).collect()
}

pub(crate) fn permute_by<const D: usize>(
    t: Tensor<D>,
    perm: &[usize],
) -> Tensor<D> {
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return t;
    }
    let arr: [isize; D] = std::array::from_fn(|i| perm[i] as isize);
    t.permute(arr)
}

pub(crate) fn to_array<T, const D: usize>(v: Vec<T>) -> [T; D] {
    v.try_into().ok().expect("length mismatch")
}

fn union_names<const D_OUT: usize>(ln: &[String], rn: &[String]) -> Vec<String> {
    let in_l = |n: &String| ln.contains(n);
    let in_r = |n: &String| rn.contains(n);

    let out: Vec<String> = ln
        .iter()
        .filter(|n| in_r(n))
        .chain(ln.iter().filter(|n| !in_r(n)))
        .chain(rn.iter().filter(|n| !in_l(n)))
        .cloned()
        .collect();
    assert_eq!(
        out.len(),
        D_OUT,
        "binary op: union has {} dims, expected D_OUT={D_OUT}",
        out.len()
    );
    out
}

macro_rules! def_binop {
    ($name:ident, $verb:expr, $op:tt) => {
        #[doc = concat!("Element-wise ", $verb, " with union broadcasting. Inputs may differ in rank.")]
        pub fn $name<const DL: usize, const DR: usize, const D_OUT: usize>(
            lhs: NamedTensor<DL>,
            rhs: NamedTensor<DR>,
        ) -> NamedTensor<D_OUT> {
            let out = union_names::<D_OUT>(&lhs.names, &rhs.names);
            let l = align::<DL, D_OUT>(lhs.inner, &lhs.names, &out);
            let r = align::<DR, D_OUT>(rhs.inner, &rhs.names, &out);
            NamedTensor::from_parts(to_array(out), l $op r)
        }
    };
}

def_binop!(add, "add", +);
def_binop!(sub, "subtract", -);
def_binop!(mul, "multiply", *);
def_binop!(div, "divide", /);

pub(crate) fn align<const DI: usize, const DO: usize>(
    t: Tensor<DI>,
    from: &[String],
    to: &[String],
) -> Tensor<DO> {
    let missing: Vec<isize> = to
        .iter()
        .enumerate()
        .filter(|(_, n)| !from.contains(n))
        .map(|(i, _)| i as isize)
        .collect();
    let expanded: Tensor<DO> = t.unsqueeze_dims(&missing);

    let mut cur = Vec::with_capacity(DO);
    let mut src = 0;
    for (i, to_name) in to.iter().enumerate().take(DO) {
        if missing.contains(&(i as isize)) {
            cur.push(to_name.clone());
        } else {
            cur.push(from[src].clone());
            src += 1;
        }
    }
    permute_by(expanded, &perm_of(&cur, to))
}

pub trait IntoContract {
    fn into_contract(self) -> Vec<String>;
}
impl IntoContract for &str {
    fn into_contract(self) -> Vec<String> {
        vec![self.to_string()]
    }
}
impl IntoContract for &[&str] {
    fn into_contract(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}
impl<const N: usize> IntoContract for [&str; N] {
    fn into_contract(self) -> Vec<String> {
        self.iter().map(|s| s.to_string()).collect()
    }
}

/// Tensor contraction over one or more named dims. Ranks may differ.
pub fn matmul<C: IntoContract, const DL: usize, const DR: usize, const D_OUT: usize>(
    lhs: NamedTensor<DL>,
    rhs: NamedTensor<DR>,
    contract: C,
) -> NamedTensor<D_OUT> {
    let ks = contract.into_contract();
    let ln: Vec<String> = lhs.names.to_vec();
    let rn: Vec<String> = rhs.names.to_vec();

    for k in &ks {
        let kl = axis_of(&ln, k);
        let kr = axis_of(&rn, k);
        assert_eq!(
            lhs.inner.shape().to_vec()[kl],
            rhs.inner.shape().to_vec()[kr],
            "matmul: K='{k}' size mismatch",
        );
    }

    let is_k = |n: &str| ks.iter().any(|k| k == n);
    let in_r = |n: &str| rn.iter().any(|x| x == n);
    let in_l = |n: &str| ln.iter().any(|x| x == n);

    let batch: Vec<String> = ln.iter().filter(|n| !is_k(n) && in_r(n)).cloned().collect();
    let m: Vec<String> = ln
        .iter()
        .filter(|n| !is_k(n) && !in_r(n))
        .cloned()
        .collect();
    let nn: Vec<String> = rn
        .iter()
        .filter(|n| !is_k(n) && !in_l(n))
        .cloned()
        .collect();

    let lhs_tgt: Vec<String> = batch
        .iter()
        .chain(m.iter())
        .chain(ks.iter())
        .cloned()
        .collect();
    let rhs_tgt: Vec<String> = batch
        .iter()
        .chain(ks.iter())
        .chain(nn.iter())
        .cloned()
        .collect();

    let lp = permute_by(lhs.inner, &perm_of(&ln, &lhs_tgt));
    let rp = permute_by(rhs.inner, &perm_of(&rn, &rhs_tgt));

    let lp_shape = lp.shape().to_vec();
    let rp_shape = rp.shape().to_vec();
    let batch_sizes: Vec<usize> = lp_shape[..batch.len()].to_vec();
    let m_sizes: Vec<usize> = lp_shape[batch.len()..batch.len() + m.len()].to_vec();
    let k_sizes: Vec<usize> = lp_shape[batch.len() + m.len()..].to_vec();
    let n_sizes: Vec<usize> = rp_shape[batch.len() + ks.len()..].to_vec();

    let prod = |xs: &[usize]| xs.iter().product::<usize>();
    let (batch_prod, m_prod, k_prod, n_prod) = (
        prod(&batch_sizes),
        prod(&m_sizes),
        prod(&k_sizes),
        prod(&n_sizes),
    );

    let lhs3: Tensor<3> = lp.reshape([batch_prod, m_prod, k_prod]);
    let rhs3: Tensor<3> = rp.reshape([batch_prod, k_prod, n_prod]);
    let raw3: Tensor<3> = lhs3.matmul(rhs3);

    let out_names: Vec<String> = batch
        .iter()
        .chain(m.iter())
        .chain(nn.iter())
        .cloned()
        .collect();
    assert_eq!(
        out_names.len(),
        D_OUT,
        "matmul: output has {} dims, expected D_OUT={D_OUT}",
        out_names.len()
    );
    let out_shape: [usize; D_OUT] = std::array::from_fn(|i| {
        if i < batch.len() {
            batch_sizes[i]
        } else if i < batch.len() + m.len() {
            m_sizes[i - batch.len()]
        } else {
            n_sizes[i - batch.len() - m.len()]
        }
    });
    let result: Tensor<D_OUT> = raw3.reshape(out_shape);

    NamedTensor::from_parts(to_array(out_names), result)
}

/// Dot product of two rank-1 tensors sharing the same dim name.
pub fn dot(lhs: NamedTensor<1>, rhs: NamedTensor<1>) -> f32 {
    assert_eq!(
        lhs.names[0], rhs.names[0],
        "dot: dim name mismatch: '{}' vs '{}'",
        lhs.names[0], rhs.names[0]
    );
    assert_eq!(
        lhs.inner.shape().to_vec()[0],
        rhs.inner.shape().to_vec()[0],
        "dot: size mismatch"
    );
    (lhs.inner * rhs.inner).sum().into_scalar::<f32>()
}

fn reduce_impl<C, const D: usize, const D_OUT: usize>(
    t: NamedTensor<D>,
    dims: C,
    mut f: impl FnMut(Tensor<D>, usize) -> Tensor<D>,
) -> NamedTensor<D_OUT>
where
    C: IntoContract,
{
    let contract = dims.into_contract();
    assert_eq!(
        D_OUT + contract.len(),
        D,
        "reduce: D_OUT ({D_OUT}) must equal D ({D}) minus contracted dims ({})",
        contract.len()
    );
    let mut inner = t.inner;
    for d in &contract {
        inner = f(inner, axis_of(&t.names, d));
    }
    let out_names: Vec<String> = t
        .names
        .iter()
        .filter(|n| !contract.contains(n))
        .cloned()
        .collect();
    let shape = inner.shape().to_vec();
    let kept: Vec<usize> = t
        .names
        .iter()
        .enumerate()
        .filter(|(_, n)| !contract.contains(n))
        .map(|(i, _)| shape[i])
        .collect();
    let out_shape: [usize; D_OUT] = std::array::from_fn(|i| kept[i]);
    let prod: usize = out_shape.iter().product::<usize>().max(1);
    let flat: Tensor<1> = inner.reshape([prod]);
    NamedTensor::from_parts(to_array(out_names), flat.reshape(out_shape))
}

macro_rules! def_reduce {
    ($name:ident, $dim_op:ident) => {
        #[doc = concat!(stringify!($dim_op), "-reduce over the given dims.")]
        pub fn $name<C: IntoContract, const D: usize, const D_OUT: usize>(
            t: NamedTensor<D>,
            dims: C,
        ) -> NamedTensor<D_OUT> {
            reduce_impl::<C, D, D_OUT>(t, dims, |inner, axis| inner.$dim_op(axis))
        }
    };
}

def_reduce!(sum, sum_dim);
def_reduce!(max, max_dim);
def_reduce!(min, min_dim);
def_reduce!(prod, prod_dim);
def_reduce!(mean, mean_dim);

/// Argmax over `dim`. Returns a plain `Tensor<D_OUT, Int>` of indices.
pub fn argmax<const D: usize, const D_OUT: usize>(
    t: NamedTensor<D>,
    dim: &str,
) -> Tensor<D_OUT, burn::tensor::Int> {
    assert_eq!(D_OUT + 1, D, "argmax: D_OUT must equal D-1");
    let axis = axis_of(&t.names, dim);
    t.inner.argmax(axis).squeeze_dim(axis)
}

/// Permute dims to `new_order`.
pub fn permute<const D: usize>(
    t: NamedTensor<D>,
    new_order: [&str; D],
) -> NamedTensor<D> {
    let from = t.names.to_vec();
    let to: Vec<String> = new_order.iter().map(|s| s.to_string()).collect();
    let fs: HashSet<&str> = from.iter().map(String::as_str).collect();
    let ts: HashSet<&str> = to.iter().map(String::as_str).collect();
    assert_eq!(
        fs, ts,
        "permute: new order {to:?} is not a permutation of {from:?}"
    );
    let inner = permute_by(t.inner, &perm_of(&from, &to));
    NamedTensor::from_parts(to_array(to), inner)
}

/// Align `t` to the target dim list `target`, permuting axes and adding size-1
/// dims for any target dim not present in `t`. Every dim of `t` must appear in
/// `target`; `target` may contain extra dims, which become size-1.
///
/// Panics if a dim of `t` is missing from `target`.
pub fn align_to<const D: usize, const D_OUT: usize>(
    t: NamedTensor<D>,
    target: [&str; D_OUT],
) -> NamedTensor<D_OUT> {
    let from = t.names.to_vec();
    let to: Vec<String> = target.iter().map(|s| s.to_string()).collect();
    for n in &from {
        assert!(
            to.contains(n),
            "align_to: dim '{n}' is not in target {to:?}",
        );
    }
    let inner = align::<D, D_OUT>(t.inner, &from, &to);
    NamedTensor::from_parts(to_array(to), inner)
}

/// Align `t` to the dim list of `other`, permuting axes and adding size-1 dims
/// for any of `other`'s dims not present in `t`. Equivalent to
/// `align_to(t, &other.names)`. `other` is borrowed only for its names.
///
/// Panics if a dim of `t` is missing from `other`.
pub fn align_as<const D: usize, const DR: usize>(
    t: NamedTensor<D>,
    other: &NamedTensor<DR>,
) -> NamedTensor<DR> {
    let target: [&str; DR] = std::array::from_fn(|i| other.names[i].as_str());
    align_to::<D, DR>(t, target)
}

/// Rename dim `old` to `new`.
pub fn rename<const D: usize>(
    mut t: NamedTensor<D>,
    old: &str,
    new: &str,
) -> NamedTensor<D> {
    let axis = axis_of(&t.names, old);
    t.names[axis] = new.to_string();
    t
}

/// Concatenate tensors along the existing dim `dim`. All inputs must share
/// the same dim *set*; each is permuted to the first tensor's order before
/// concatenation, so differing axis orders are tolerated. The output keeps
/// the first tensor's dim order.
///
/// Panics if `tensors` is empty, if a tensor is missing a dim, or if any
/// non-concat dim has mismatched sizes across inputs.
pub fn concat<const D: usize>(
    tensors: Vec<NamedTensor<D>>,
    dim: &str,
) -> NamedTensor<D> {
    let target = tensors[0].names.clone();
    let axis = axis_of(&target, dim);
    let inners: Vec<Tensor<D>> = tensors
        .into_iter()
        .map(|t| permute_by(t.inner, &perm_of(&t.names, &target)))
        .collect();
    NamedTensor::from_parts(target, Tensor::cat(inners, axis))
}

/// Stack tensors along a *new* dim `dim`, prepended to the front of the dim
/// list. All inputs must share the same dim *set*; each is permuted to the
/// first tensor's order first. `D_OUT` must equal `D + 1`.
///
/// Panics if `tensors` is empty, if a tensor is missing a dim, or if the
/// inputs' shapes differ.
pub fn stack<const D: usize, const D_OUT: usize>(
    tensors: Vec<NamedTensor<D>>,
    dim: &str,
) -> NamedTensor<D_OUT> {
    debug_assert_eq!(D_OUT, D + 1, "stack: D_OUT must equal D + 1");
    let target = tensors[0].names.clone();
    let inners: Vec<Tensor<D>> = tensors
        .into_iter()
        .map(|t| permute_by(t.inner, &perm_of(&t.names, &target)))
        .collect();
    let mut names = vec![dim.to_string()];
    names.extend(target);
    NamedTensor::from_parts(to_array(names), Tensor::stack::<D_OUT>(inners, 0))
}
