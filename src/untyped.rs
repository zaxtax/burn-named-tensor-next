//! Runtime-checked named tensors — the untyped counterpart to [`crate::typed`].
//!
//! Dim names are `String`s carried alongside a [`burn::tensor::Tensor`], and all
//! shape/dim constraints are checked at run time. The API mirrors [`crate::typed`]
//! so code can be ported by swapping type-level dim markers for string literals.

use burn::prelude::*;
use burn::tensor::Shape;
use std::collections::HashSet;
use std::marker::PhantomData;

/// A burn tensor with a `String` name per axis.
pub struct NamedTensor<B: Backend, const D: usize> {
    pub inner: Tensor<B, D>,
    pub names: [String; D],
    _b: PhantomData<fn() -> B>,
}

impl<B: Backend, const D: usize> NamedTensor<B, D> {
    pub fn new(names: [&str; D], inner: Tensor<B, D>) -> Self {
        Self { inner, names: names.map(String::from), _b: PhantomData }
    }

    pub fn from_parts(names: [String; D], inner: Tensor<B, D>) -> Self {
        Self { inner, names, _b: PhantomData }
    }

    pub fn into_inner(self) -> Tensor<B, D> { self.inner }
    pub fn shape(&self) -> Shape { self.inner.shape() }
    pub fn names(&self) -> &[String; D] { &self.names }
}

impl<B: Backend, const D: usize> Clone for NamedTensor<B, D>
where Tensor<B, D>: Clone {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), names: self.names.clone(), _b: PhantomData }
    }
}
impl<B: Backend, const D: usize> std::fmt::Debug for NamedTensor<B, D>
where Tensor<B, D>: std::fmt::Debug {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedTensor")
            .field("names", &self.names)
            .field("inner", &self.inner)
            .finish()
    }
}
impl<B: Backend, const D: usize> std::fmt::Display for NamedTensor<B, D>
where Tensor<B, D>: std::fmt::Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} ", self.names)?;
        self.inner.fmt(f)
    }
}

// Runtime axis utilities

fn find_axis(list: &[String], name: &str) -> usize {
    list.iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("named-tensor: dim '{name}' not found in {list:?}"))
}

fn build_perm(from: &[String], to: &[String]) -> Vec<usize> {
    to.iter().map(|n| find_axis(from, n)).collect()
}

fn is_identity(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

fn permute_if_needed<B: Backend, const D: usize>(
    t: Tensor<B, D>,
    perm: &[usize],
) -> Tensor<B, D> {
    if is_identity(perm) { return t; }
    let arr: [isize; D] = perm.iter().map(|&x| x as isize)
        .collect::<Vec<_>>().try_into().unwrap();
    t.permute(arr)
}

fn contains_name(list: &[String], name: &str) -> bool {
    list.iter().any(|n| n == name)
}

// Operations

/// Element-wise add with union broadcasting. Inputs may differ in rank.
///
/// Output dim order: shared dims (in `lhs` order), then `lhs`-only, then `rhs`-only —
/// matching the Harvard NLP NamedTensor convention. `D_OUT` must equal the size of
/// the union; this is checked at run time.
pub fn add<B, const DL: usize, const DR: usize, const D_OUT: usize>(
    lhs: NamedTensor<B, DL>,
    rhs: NamedTensor<B, DR>,
) -> NamedTensor<B, D_OUT>
where
    B: Backend,
{
    let lhs_names = lhs.names.to_vec();
    let rhs_names = rhs.names.to_vec();

    let mut out_names: Vec<String> = Vec::new();
    for n in &lhs_names {
        if contains_name(&rhs_names, n) { out_names.push(n.clone()); }
    }
    for n in &lhs_names {
        if !contains_name(&rhs_names, n) { out_names.push(n.clone()); }
    }
    for n in &rhs_names {
        if !contains_name(&lhs_names, n) { out_names.push(n.clone()); }
    }
    assert_eq!(
        out_names.len(), D_OUT,
        "add: union has {} dims, expected D_OUT={}", out_names.len(), D_OUT,
    );

    let l = align::<B, DL, D_OUT>(lhs.inner, &lhs_names, &out_names);
    let r = align::<B, DR, D_OUT>(rhs.inner, &rhs_names, &out_names);

    let names_arr: [String; D_OUT] = out_names.try_into().unwrap();
    NamedTensor::from_parts(names_arr, l + r)
}

fn align<B: Backend, const D_IN: usize, const D_OUT: usize>(
    t: Tensor<B, D_IN>,
    operand_names: &[String],
    out_names: &[String],
) -> Tensor<B, D_OUT> {
    let missing: Vec<isize> = out_names.iter().enumerate()
        .filter(|(_, n)| !contains_name(operand_names, n))
        .map(|(i, _)| i as isize)
        .collect();

    let expanded: Tensor<B, D_OUT> = t.unsqueeze_dims(&missing);

    let mut current: Vec<String> = vec![String::new(); D_OUT];
    let mut src = 0usize;
    for i in 0..D_OUT {
        if missing.contains(&(i as isize)) {
            current[i] = out_names[i].clone();
        } else {
            current[i] = operand_names[src].clone();
            src += 1;
        }
    }

    let perm = build_perm(&current, out_names);
    permute_if_needed(expanded, &perm)
}

/// Matrix multiply contracting over dim `contract`. May appear at any axis on either side.
/// Both tensors must share the same burn rank `D`, and `D == D_OUT` (mirrors the typed API).
///
/// Output dim order: batch dims (lhs order), then lhs-only M-dims, then rhs-only N-dims.
pub fn matmul<B, const D: usize, const D_OUT: usize>(
    lhs: NamedTensor<B, D>,
    rhs: NamedTensor<B, D>,
    contract: &str,
) -> NamedTensor<B, D_OUT>
where
    B: Backend,
{
    assert_eq!(D, D_OUT, "matmul: D={D} ≠ D_OUT={D_OUT}");

    let lhs_names = lhs.names.to_vec();
    let rhs_names = rhs.names.to_vec();
    let k = contract;

    let k_l = find_axis(&lhs_names, k);
    let k_r = find_axis(&rhs_names, k);
    assert_eq!(
        lhs.inner.shape().dims[k_l], rhs.inner.shape().dims[k_r],
        "matmul: K='{k}' size mismatch",
    );

    let rhs_set: HashSet<&str> = rhs_names.iter().map(|s| s.as_str()).collect();
    let lhs_set: HashSet<&str> = lhs_names.iter().map(|s| s.as_str()).collect();

    let batch_names: Vec<String> = lhs_names.iter()
        .filter(|n| n.as_str() != k && rhs_set.contains(n.as_str()))
        .cloned().collect();
    let m_names: Vec<String> = lhs_names.iter()
        .filter(|n| n.as_str() != k && !rhs_set.contains(n.as_str()))
        .cloned().collect();
    let n_names: Vec<String> = rhs_names.iter()
        .filter(|n| n.as_str() != k && !lhs_set.contains(n.as_str()))
        .cloned().collect();

    let k_owned = k.to_string();
    let lhs_target: Vec<String> = batch_names.iter()
        .chain(m_names.iter())
        .chain(std::iter::once(&k_owned))
        .cloned().collect();
    let rhs_target: Vec<String> = batch_names.iter()
        .chain(std::iter::once(&k_owned))
        .chain(n_names.iter())
        .cloned().collect();

    let lhs_p = permute_if_needed(lhs.inner, &build_perm(&lhs_names, &lhs_target));
    let rhs_p = permute_if_needed(rhs.inner, &build_perm(&rhs_names, &rhs_target));

    let raw: Tensor<B, D> = lhs_p.matmul(rhs_p);
    let raw_names: Vec<String> = batch_names.into_iter()
        .chain(m_names.into_iter())
        .chain(n_names.into_iter())
        .collect();
    assert_eq!(
        raw_names.len(), D_OUT,
        "matmul: output has {} dims, expected D_OUT={}", raw_names.len(), D_OUT,
    );

    // SAFETY: D == D_OUT (checked above); Tensor<B,D> and Tensor<B,D_OUT> are layout-identical.
    let result: Tensor<B, D_OUT> = unsafe {
        std::mem::transmute_copy(&std::mem::ManuallyDrop::new(raw))
    };
    let names_arr: [String; D_OUT] = raw_names.try_into().unwrap();
    NamedTensor::from_parts(names_arr, result)
}

/// Dot product of two rank-1 tensors sharing the same dim.
pub fn dot<B>(lhs: NamedTensor<B, 1>, rhs: NamedTensor<B, 1>) -> f32
where
    B: Backend,
    B::FloatElem: Into<f32>,
{
    assert_eq!(
        lhs.names[0], rhs.names[0],
        "dot: dim name mismatch: '{}' vs '{}'", lhs.names[0], rhs.names[0],
    );
    assert_eq!(
        lhs.inner.shape().dims[0], rhs.inner.shape().dims[0],
        "dot: size mismatch",
    );
    (lhs.inner * rhs.inner).sum().into_scalar().into()
}

/// Sum-reduce over the named dim `dim`. `D_OUT` must equal `D - 1`.
pub fn sum<B, const D: usize, const D_OUT: usize>(
    t: NamedTensor<B, D>,
    dim: &str,
) -> NamedTensor<B, D_OUT>
where
    B: Backend,
{
    assert_eq!(D_OUT + 1, D, "sum: D_OUT must equal D-1");
    let names = t.names.to_vec();
    let axis = find_axis(&names, dim);

    let squeezed: Tensor<B, D_OUT> = t.inner.sum_dim(axis).squeeze_dim(axis);

    let mut out_names = names;
    out_names.remove(axis);
    let names_arr: [String; D_OUT] = out_names.try_into().unwrap();
    NamedTensor::from_parts(names_arr, squeezed)
}

/// Permute (transpose) dims to `new_order` — like xarray's `.transpose()`.
/// `new_order` must be a permutation of the tensor's current dim names.
pub fn permute<B, const D: usize>(
    t: NamedTensor<B, D>,
    new_order: [&str; D],
) -> NamedTensor<B, D>
where
    B: Backend,
{
    let from = t.names.to_vec();
    let to: Vec<String> = new_order.iter().map(|s| s.to_string()).collect();

    {
        let from_set: HashSet<&str> = from.iter().map(|s| s.as_str()).collect();
        let to_set: HashSet<&str> = to.iter().map(|s| s.as_str()).collect();
        assert_eq!(
            from_set, to_set,
            "permute: new order {:?} is not a permutation of {:?}", to, from,
        );
    }

    let perm = build_perm(&from, &to);
    let inner = permute_if_needed(t.inner, &perm);
    let names_arr: [String; D] = to.try_into().unwrap();
    NamedTensor::from_parts(names_arr, inner)
}

/// Rename dim `old` to `new` — zero cost, no data movement. Panics if `old` is absent.
pub fn rename<B, const D: usize>(
    mut t: NamedTensor<B, D>,
    old: &str,
    new: &str,
) -> NamedTensor<B, D>
where
    B: Backend,
{
    let axis = find_axis(&t.names.as_slice().to_vec(), old);
    t.names[axis] = new.to_string();
    t
}
