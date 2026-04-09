//! Runtime-checked named tensors — the untyped counterpart to [`crate::typed`].
//!
//! Dim names are `String`s carried alongside a [`burn::tensor::Tensor`], and all
//! shape/dim constraints are checked at run time.

use burn::prelude::*;
use burn::tensor::Shape;
use std::collections::HashSet;

/// A burn tensor with a `String` name per axis.
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

// ---------- axis helpers ----------

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

// ---------- operations ----------

/// Element-wise add with union broadcasting. Inputs may differ in rank.
///
/// Output order: shared dims (lhs order), then lhs-only, then rhs-only.
/// `D_OUT` must equal the size of the union.
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

/// Matrix multiply contracting over dim `contract`. Both sides share rank `D`,
/// and `D == D_OUT`. Output order: batch dims (lhs order), then lhs-only, then rhs-only.
pub fn matmul<B: Backend, const D: usize, const D_OUT: usize>(
    lhs: NamedTensor<B, D>,
    rhs: NamedTensor<B, D>,
    contract: &str,
) -> NamedTensor<B, D_OUT> {
    assert_eq!(D, D_OUT, "matmul: D={D} ≠ D_OUT={D_OUT}");
    let (ln, rn) = (lhs.names.to_vec(), rhs.names.to_vec());
    let (kl, kr) = (axis_of(&ln, contract), axis_of(&rn, contract));
    assert_eq!(
        lhs.inner.shape().dims[kl],
        rhs.inner.shape().dims[kr],
        "matmul: K='{contract}' size mismatch",
    );

    let lset: HashSet<&str> = ln.iter().map(String::as_str).collect();
    let rset: HashSet<&str> = rn.iter().map(String::as_str).collect();
    let not_k = |n: &&String| n.as_str() != contract;
    let batch: Vec<String> = ln
        .iter()
        .filter(not_k)
        .filter(|n| rset.contains(n.as_str()))
        .cloned()
        .collect();
    let m: Vec<String> = ln
        .iter()
        .filter(not_k)
        .filter(|n| !rset.contains(n.as_str()))
        .cloned()
        .collect();
    let nn: Vec<String> = rn
        .iter()
        .filter(not_k)
        .filter(|n| !lset.contains(n.as_str()))
        .cloned()
        .collect();

    let k = contract.to_string();
    let lhs_tgt: Vec<String> = batch
        .iter()
        .chain(m.iter())
        .chain(std::iter::once(&k))
        .cloned()
        .collect();
    let rhs_tgt: Vec<String> = batch
        .iter()
        .chain(std::iter::once(&k))
        .chain(nn.iter())
        .cloned()
        .collect();

    let lp = permute_by(lhs.inner, &perm_of(&ln, &lhs_tgt));
    let rp = permute_by(rhs.inner, &perm_of(&rn, &rhs_tgt));
    let raw: Tensor<B, D> = lp.matmul(rp);

    let out: Vec<String> = batch.into_iter().chain(m).chain(nn).collect();
    assert_eq!(
        out.len(),
        D_OUT,
        "matmul: output has {} dims, expected D_OUT={D_OUT}",
        out.len()
    );

    // SAFETY: D == D_OUT (asserted above); Tensor<B,D> and Tensor<B,D_OUT> are layout-identical.
    let result: Tensor<B, D_OUT> =
        unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(raw)) };
    NamedTensor::from_parts(to_array(out), result)
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

/// Sum-reduce over `dim`. `D_OUT` must equal `D - 1`.
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

/// Permute dims to `new_order` — like xarray's `.transpose()`.
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

/// Rename dim `old` to `new` — zero cost. Panics if `old` is absent.
pub fn rename<B: Backend, const D: usize>(
    mut t: NamedTensor<B, D>,
    old: &str,
    new: &str,
) -> NamedTensor<B, D> {
    let axis = axis_of(&t.names, old);
    t.names[axis] = new.to_string();
    t
}
