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

pub(crate) fn permute_by<B: Backend, const D: usize>(
    t: Tensor<B, D>,
    perm: &[usize],
) -> Tensor<B, D> {
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
        pub fn $name<B: Backend, const DL: usize, const DR: usize, const D_OUT: usize>(
            lhs: NamedTensor<B, DL>,
            rhs: NamedTensor<B, DR>,
        ) -> NamedTensor<B, D_OUT> {
            let out = union_names::<D_OUT>(&lhs.names, &rhs.names);
            let l = align::<B, DL, D_OUT>(lhs.inner, &lhs.names, &out);
            let r = align::<B, DR, D_OUT>(rhs.inner, &rhs.names, &out);
            NamedTensor::from_parts(to_array(out), l $op r)
        }
    };
}

def_binop!(add, "add", +);
def_binop!(sub, "subtract", -);
def_binop!(mul, "multiply", *);
def_binop!(div, "divide", /);

pub(crate) fn align<B: Backend, const DI: usize, const DO: usize>(
    t: Tensor<B, DI>,
    from: &[String],
    to: &[String],
) -> Tensor<B, DO> {
    let missing: Vec<isize> = to
        .iter()
        .enumerate()
        .filter(|(_, n)| !from.contains(n))
        .map(|(i, _)| i as isize)
        .collect();
    let expanded: Tensor<B, DO> = t.unsqueeze_dims(&missing);

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
pub fn matmul<B: Backend, C: IntoContract, const DL: usize, const DR: usize, const D_OUT: usize>(
    lhs: NamedTensor<B, DL>,
    rhs: NamedTensor<B, DR>,
    contract: C,
) -> NamedTensor<B, D_OUT> {
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

    let lhs3: Tensor<B, 3> = lp.reshape([batch_prod, m_prod, k_prod]);
    let rhs3: Tensor<B, 3> = rp.reshape([batch_prod, k_prod, n_prod]);
    let raw3: Tensor<B, 3> = lhs3.matmul(rhs3);

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
    let result: Tensor<B, D_OUT> = raw3.reshape(out_shape);

    NamedTensor::from_parts(to_array(out_names), result)
}

/// Dot product of two rank-1 tensors sharing the same dim name.
pub fn dot<B: Backend>(lhs: NamedTensor<B, 1>, rhs: NamedTensor<B, 1>) -> f32
where
    B::FloatElem: Into<f32>,
{
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
    (lhs.inner * rhs.inner).sum().into_scalar().into()
}

fn reduce_impl<B, C, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, D>,
    dims: C,
    mut f: impl FnMut(Tensor<B, D>, usize) -> Tensor<B, D>,
) -> NamedTensor<B, D_OUT>
where
    B: Backend,
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
    let flat: Tensor<B, 1> = inner.reshape([prod]);
    NamedTensor::from_parts(to_array(out_names), flat.reshape(out_shape))
}

macro_rules! def_reduce {
    ($name:ident, $dim_op:ident) => {
        #[doc = concat!(stringify!($dim_op), "-reduce over the given dims.")]
        pub fn $name<B: Backend, C: IntoContract, const D: usize, const D_OUT: usize>(
            t: NamedTensor<B, D>,
            dims: C,
        ) -> NamedTensor<B, D_OUT> {
            reduce_impl::<B, C, D, D_OUT>(t, dims, |inner, axis| inner.$dim_op(axis))
        }
    };
}

def_reduce!(sum, sum_dim);
def_reduce!(max, max_dim);
def_reduce!(min, min_dim);
def_reduce!(prod, prod_dim);
def_reduce!(mean, mean_dim);

/// Argmax over `dim`. Returns a plain `Tensor<B, D_OUT, Int>` of indices.
pub fn argmax<B: Backend, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, D>,
    dim: &str,
) -> Tensor<B, D_OUT, burn::tensor::Int> {
    assert_eq!(D_OUT + 1, D, "argmax: D_OUT must equal D-1");
    let axis = axis_of(&t.names, dim);
    t.inner.argmax(axis).squeeze_dim(axis)
}

/// Permute dims to `new_order`.
pub fn permute<B: Backend, const D: usize>(
    t: NamedTensor<B, D>,
    new_order: [&str; D],
) -> NamedTensor<B, D> {
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

/// Rename dim `old` to `new`.
pub fn rename<B: Backend, const D: usize>(
    mut t: NamedTensor<B, D>,
    old: &str,
    new: &str,
) -> NamedTensor<B, D> {
    let axis = axis_of(&t.names, old);
    t.names[axis] = new.to_string();
    t
}
