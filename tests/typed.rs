#![allow(clippy::type_complexity)]

use burn::backend::NdArray;
use burn::tensor::{Shape, Tensor, TensorData};
use named_tensor::typed::{add, div, dot, matmul, mul, permute, rename, sub, sum, NamedTensor};
use named_tensor::{dim, dims};

dim!(Batch, M, K, N, Features, SeqLen, Hidden);

type B = NdArray<f32>;

fn dev() -> <B as burn::prelude::Backend>::Device {
    Default::default()
}

#[test]
fn add_same_shape() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev));
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let c: NamedTensor<B, dims![M, N], 2> = add(a, b);
    assert_eq!(c.dim_names(), &["M", "N"]);
    assert_eq!(c.shape().dims, [3, 5]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn add_with_plus_operator() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev));
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let c = a + b;
    assert_eq!(c.dim_names(), &["M", "N"]);
    assert_eq!(c.shape().dims, [3, 5]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn add_rank2_rank1_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new((1..=15).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 5]),
        &dev,
    ));
    let bias: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([0.1f32, 0.2, 0.3, 0.4, 0.5], &dev));
    let out: NamedTensor<B, dims![M, N], 2> = add(mat, bias);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().dims, [3, 5]);
}

#[test]
fn add_disjoint_dims() {
    let dev = dev();
    let row: NamedTensor<B, dims![M], 1> =
        NamedTensor::new(Tensor::from_data([1.0f32, 2.0, 3.0], &dev));
    let col: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([10.0f32, 20.0, 30.0, 40.0, 50.0], &dev));
    let out: NamedTensor<B, dims![M, N], 2> = add(row, col);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().dims, [3, 5]);
}

#[test]
fn add_commuted_order() {
    let dev = dev();
    let bias: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([1.0f32, 1.0, 1.0, 1.0, 1.0], &dev));
    let mat: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let out: NamedTensor<B, dims![N, M], 2> = add(bias, mat);
    assert_eq!(out.dim_names(), &["N", "M"]);
    assert_eq!(out.shape().dims, [5, 3]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn matmul_2d_standard() {
    let dev = dev();
    let lhs_data: Vec<f32> = (0..3).flat_map(|r| vec![(r + 1) as f32; 4]).collect();
    let lhs: NamedTensor<B, dims![M, K], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(lhs_data, [3usize, 4]),
        &dev,
    ));
    let rhs_data: Vec<f32> = (0..4)
        .flat_map(|_| (1..=5).map(|c| c as f32 * 0.1).collect::<Vec<_>>())
        .collect();
    let rhs: NamedTensor<B, dims![K, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(rhs_data, [4usize, 5]),
        &dev,
    ));
    let c: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs, K);
    assert_eq!(c.dim_names(), &["M", "N"]);
    assert_eq!(c.shape().dims, [3, 5]);
}

#[test]
fn matmul_2d_k_nonstandard() {
    let dev = dev();
    let lhs: NamedTensor<B, dims![K, M], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new((1..=12).map(|x| x as f32).collect::<Vec<_>>(), [4usize, 3]),
        &dev,
    ));
    let rhs: NamedTensor<B, dims![N, K], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=20).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
            [5usize, 4],
        ),
        &dev,
    ));
    let c: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs, K);
    assert_eq!(c.dim_names(), &["M", "N"]);
    assert_eq!(c.shape().dims, [3, 5]);
}

#[test]
fn matmul_3d_batched() {
    let dev = dev();
    let lhs: NamedTensor<B, dims![Batch, M, K], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
            [2usize, 3, 4],
        ),
        &dev,
    ));
    let rhs: NamedTensor<B, dims![Batch, K, N], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
            [2usize, 4, 5],
        ),
        &dev,
    ));
    let out: NamedTensor<B, dims![Batch, M, N], 3> = matmul(lhs, rhs, K);
    assert_eq!(out.dim_names(), &["Batch", "M", "N"]);
    assert_eq!(out.shape().dims, [2, 3, 5]);
}

#[test]
fn matmul_3d_k_middle() {
    let dev = dev();
    let lhs: NamedTensor<B, dims![M, K, Batch], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
            [3usize, 4, 2],
        ),
        &dev,
    ));
    let rhs: NamedTensor<B, dims![Batch, K, N], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
            [2usize, 4, 5],
        ),
        &dev,
    ));
    let out: NamedTensor<B, dims![M, Batch, N], 3> = matmul(lhs, rhs, K);
    assert_eq!(out.dim_names(), &["M", "Batch", "N"]);
    assert_eq!(out.shape().dims, [3, 2, 5]);
}

#[test]
fn matmul_mixed_rank() {
    let dev = dev();
    let lhs: NamedTensor<B, dims![M, K], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new((1..=6).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 2]),
        &dev,
    ));
    let rhs: NamedTensor<B, dims![K, N, Batch], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
            [2usize, 5, 4],
        ),
        &dev,
    ));
    let out: NamedTensor<B, dims![M, N, Batch], 3> = matmul(lhs, rhs, K);
    assert_eq!(out.dim_names(), &["M", "N", "Batch"]);
    assert_eq!(out.shape().dims, [3, 5, 4]);
}

#[test]
fn dot_scalar() {
    let dev = dev();
    let u: NamedTensor<B, dims![Features], 1> =
        NamedTensor::new(Tensor::from_data([1.0f32, 2.0, 3.0, 4.0], &dev));
    let v: NamedTensor<B, dims![Features], 1> =
        NamedTensor::new(Tensor::from_data([0.25f32, 0.5, 0.75, 1.0], &dev));
    let s: f32 = dot(u, v);
    assert!((s - 7.5).abs() < 1e-4, "expected 7.5, got {s}");
}

#[test]
fn dot_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![Batch, Features], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            vec![
                1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
            ],
            [3usize, 4],
        ),
        &dev,
    ));
    let bias: NamedTensor<B, dims![Features], 1> =
        NamedTensor::new(Tensor::from_data([1.0f32, 2.0, 3.0, 4.0], &dev));
    let out: NamedTensor<B, dims![Batch], 1> = dot(mat, bias);
    assert_eq!(out.dim_names(), &["Batch"]);
    assert_eq!(out.shape().dims, [3]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 20.0).abs() < 1e-4, "expected mean 20.0, got {mean}");
}

#[test]
fn permute_dims() {
    let dev = dev();
    let t: NamedTensor<B, dims![Batch, M, N], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=30).map(|x| x as f32).collect::<Vec<_>>(),
            [2usize, 3, 5],
        ),
        &dev,
    ));
    assert_eq!(t.shape().dims, [2, 3, 5]);
    let t2: NamedTensor<B, dims![N, Batch, M], 3> = permute(t);
    assert_eq!(t2.dim_names(), &["N", "Batch", "M"]);
    assert_eq!(t2.shape().dims, [5, 2, 3]);
}

#[test]
fn sum_and_rename() {
    let dev = dev();
    let t: NamedTensor<B, dims![SeqLen, Features], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([4usize, 8]), &dev));
    let s: NamedTensor<B, dims![Features], 1> = sum::<B, SeqLen, _, _, _, 2, 1>(t);
    assert_eq!(s.dim_names(), &["Features"]);
    assert_eq!(s.shape().dims, [8]);
    let h: NamedTensor<B, dims![Hidden], 1> = rename::<B, Features, Hidden, _, _, _, 1>(s);
    assert_eq!(h.dim_names(), &["Hidden"]);
}

#[test]
fn sum_to_scalar() {
    let dev = dev();
    let t: NamedTensor<B, dims![SeqLen, Features], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([4usize, 8]), &dev));
    let s: NamedTensor<B, dims![Features], 1> = sum::<B, SeqLen, _, _, _, 2, 1>(t);
    let h: NamedTensor<B, dims![Hidden], 1> = rename::<B, Features, Hidden, _, _, _, 1>(s);
    let total: f32 = sum::<B, Hidden, _, _, _, 1, 0>(h);
    assert!((total - 32.0).abs() < 1e-4, "expected 32.0, got {total}");
}

#[test]
fn mean_reduce() {
    let dev = dev();
    let t: NamedTensor<B, dims![SeqLen, Features], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2usize, 4],
        ),
        &dev,
    ));
    let m: NamedTensor<B, dims![Features], 1> = t.mean::<dims![SeqLen], _, _, 1>();
    assert_eq!(m.dim_names(), &["Features"]);
    assert_eq!(m.shape().dims, [4]);
    let val: f32 = m.inner.mean().into_scalar();
    assert!((val - 4.5).abs() < 1e-4, "expected 4.5, got {val}");
}

#[test]
fn mean_to_scalar() {
    let dev = dev();
    let t: NamedTensor<B, dims![Features], 1> =
        NamedTensor::new(Tensor::from_data([2.0f32, 4.0, 6.0, 8.0], &dev));
    let s: f32 = t.mean::<dims![Features], _, _, 0>();
    assert!((s - 5.0).abs() < 1e-4, "expected 5.0, got {s}");
}

#[test]
fn mean_multi_dim() {
    let dev = dev();
    let t: NamedTensor<B, dims![SeqLen, Features], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2usize, 4],
        ),
        &dev,
    ));
    let s: f32 = t.mean::<dims![SeqLen, Features], _, _, 0>();
    assert!((s - 4.5).abs() < 1e-4, "expected 4.5, got {s}");
}

#[test]
fn mean_multi_dim_partial() {
    let dev = dev();
    let t: NamedTensor<B, dims![Batch, SeqLen, Features], 3> =
        NamedTensor::new(Tensor::from_data(
            TensorData::new(
                (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                [2usize, 3, 4],
            ),
            &dev,
        ));
    let m: NamedTensor<B, dims![Features], 1> =
        t.mean::<dims![Batch, SeqLen], _, _, 1>();
    assert_eq!(m.dim_names(), &["Features"]);
    assert_eq!(m.shape().dims, [4]);
    // mean over Batch and SeqLen for each of the 4 features
    let val: f32 = m.inner.mean().into_scalar();
    assert!((val - 12.5).abs() < 1e-4, "expected 12.5, got {val}");
}

// ── sub ──

#[test]
fn sub_same_shape() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 5.0);
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let c: NamedTensor<B, dims![M, N], 2> = sub(a, b);
    assert_eq!(c.dim_names(), &["M", "N"]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn sub_with_minus_operator() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 5.0);
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let c = a - b;
    assert_eq!(c.dim_names(), &["M", "N"]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn sub_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 10.0);
    let bias: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0], &dev));
    let out: NamedTensor<B, dims![M, N], 2> = sub(mat, bias);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().dims, [3, 5]);
}

// ── mul ──

#[test]
fn mul_same_shape() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 3.0);
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let c: NamedTensor<B, dims![M, N], 2> = mul(a, b);
    assert_eq!(c.dim_names(), &["M", "N"]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 6.0).abs() < 1e-4, "expected mean 6.0, got {mean}");
}

#[test]
fn mul_with_star_operator() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 3.0);
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let c = a * b;
    assert_eq!(c.dim_names(), &["M", "N"]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 6.0).abs() < 1e-4, "expected mean 6.0, got {mean}");
}

#[test]
fn mul_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 4.0);
    let scale: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([2.0f32, 2.0, 2.0, 2.0, 2.0], &dev));
    let out: NamedTensor<B, dims![M, N], 2> = mul(mat, scale);
    assert_eq!(out.dim_names(), &["M", "N"]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 8.0).abs() < 1e-4, "expected mean 8.0, got {mean}");
}

// ── div ──

#[test]
fn div_same_shape() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 6.0);
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let c: NamedTensor<B, dims![M, N], 2> = div(a, b);
    assert_eq!(c.dim_names(), &["M", "N"]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn div_with_slash_operator() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 6.0);
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
    let c = a / b;
    assert_eq!(c.dim_names(), &["M", "N"]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn div_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 10.0);
    let scale: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([2.0f32, 2.0, 2.0, 2.0, 2.0], &dev));
    let out: NamedTensor<B, dims![M, N], 2> = div(mat, scale);
    assert_eq!(out.dim_names(), &["M", "N"]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 5.0).abs() < 1e-4, "expected mean 5.0, got {mean}");
}

// ── operator cross-rank broadcast ──

#[test]
fn add_operator_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev));
    let bias: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0], &dev));
    let out = mat + bias;
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().dims, [3, 5]);
}

#[test]
fn sub_operator_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 10.0);
    let bias: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0], &dev));
    let out = mat - bias;
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().dims, [3, 5]);
}

#[test]
fn mul_operator_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 4.0);
    let scale: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([2.0f32, 2.0, 2.0, 2.0, 2.0], &dev));
    let out = mat * scale;
    assert_eq!(out.dim_names(), &["M", "N"]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 8.0).abs() < 1e-4, "expected mean 8.0, got {mean}");
}

#[test]
fn div_operator_broadcast() {
    let dev = dev();
    let mat: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 10.0);
    let scale: NamedTensor<B, dims![N], 1> =
        NamedTensor::new(Tensor::from_data([2.0f32, 2.0, 2.0, 2.0, 2.0], &dev));
    let out = mat / scale;
    assert_eq!(out.dim_names(), &["M", "N"]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 5.0).abs() < 1e-4, "expected mean 5.0, got {mean}");
}

// ── untyped roundtrip ──

#[test]
fn untyped_roundtrip() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev) * 3.0);

    let u = a.untyped();
    assert_eq!(u.names(), &["M".to_string(), "N".to_string()]);

    let back: NamedTensor<B, dims![M, N], 2> = u.to_named();
    assert_eq!(back.dim_names(), &["M", "N"]);
    assert_eq!(back.shape().dims, [3, 5]);
    let mean: f32 = back.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4);
}

#[test]
fn untyped_roundtrip_permuted() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ));

    let u = a.untyped();
    let back: NamedTensor<B, dims![N, M], 2> = u.to_named();
    assert_eq!(back.dim_names(), &["N", "M"]);
    assert_eq!(back.shape().dims, [3, 2]);
}
