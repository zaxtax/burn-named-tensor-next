//! Runtime-checked named tensors — the untyped counterpart to [`crate::typed`].

use burn::prelude::*;
use burn::tensor::Shape;
use std::collections::HashSet;
use std::ops::Add;

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
    pub fn into_inner(self) -> Tensor<B, D> {
        self.inner
    }
    pub fn shape(&self) -> Shape {
        self.inner.shape()
    }
    pub fn names(&self) -> &[String; D] {
        &self.names
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

impl<B: Backend, const D: usize> Add for NamedTensor<B, D> {
    type Output = NamedTensor<B, D>;

    fn add(self, rhs: NamedTensor<B, D>) -> Self::Output {
        NamedTensor::from_parts(self.names, self.inner + rhs.inner)
    }
}

fn axis_of(names: &[String], name: &str) -> usize {
    names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("named-tensor: dim '{name}' not found in {names:?}"))
}

fn perm_of(from: &[String], to: &[String]) -> Vec<usize> {
    to.iter().map(|n| axis_of(from, n)).collect()
}

fn permute_by<B: Backend, const D: usize>(t: Tensor<B, D>, perm: &[usize]) -> Tensor<B, D> {
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return t;
    }
    let arr: [isize; D] = std::array::from_fn(|i| perm[i] as isize);
    t.permute(arr)
}

fn to_array<T, const D: usize>(v: Vec<T>) -> [T; D] {
    v.try_into().ok().expect("length mismatch")
}

/// Element-wise add with union broadcasting. Inputs may differ in rank.
pub fn add<B: Backend, const DL: usize, const DR: usize, const D_OUT: usize>(
    lhs: NamedTensor<B, DL>,
    rhs: NamedTensor<B, DR>,
) -> NamedTensor<B, D_OUT> {
    let (ln, rn) = (lhs.names.to_vec(), rhs.names.to_vec());
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
        "add: union has {} dims, expected D_OUT={D_OUT}",
        out.len()
    );

    let l = align::<B, DL, D_OUT>(lhs.inner, &ln, &out);
    let r = align::<B, DR, D_OUT>(rhs.inner, &rn, &out);
    NamedTensor::from_parts(to_array(out), l + r)
}

fn align<B: Backend, const DI: usize, const DO: usize>(
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
    for i in 0..DO {
        if missing.contains(&(i as isize)) {
            cur.push(to[i].clone());
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
pub fn matmul<
    B: Backend,
    C: IntoContract,
    const DL: usize,
    const DR: usize,
    const D_OUT: usize,
>(
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
            lhs.inner.shape().dims[kl],
            rhs.inner.shape().dims[kr],
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

    let lp_shape = lp.shape().dims;
    let rp_shape = rp.shape().dims;
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
        lhs.inner.shape().dims[0],
        rhs.inner.shape().dims[0],
        "dot: size mismatch"
    );
    (lhs.inner * rhs.inner).sum().into_scalar().into()
}

/// Sum-reduce over `dim`.
pub fn sum<B: Backend, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, D>,
    dim: &str,
) -> NamedTensor<B, D_OUT> {
    assert_eq!(D_OUT + 1, D, "sum: D_OUT must equal D-1");
    let axis = axis_of(&t.names, dim);
    let reduced: Tensor<B, D_OUT> = t.inner.sum_dim(axis).squeeze_dim(axis);
    let mut names = t.names.to_vec();
    names.remove(axis);
    NamedTensor::from_parts(to_array(names), reduced)
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
