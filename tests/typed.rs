#![allow(clippy::type_complexity)]

use burn::backend::Flex;
use burn::tensor::{Shape, Tensor, TensorData};
use named_tensor::typed::{
    NamedTensor, add, align_as, align_to, concat, div, dot, matmul, mul, permute, rename, stack,
    sub, sum,
};
use named_tensor::{dim, dims, s};

dim!(Batch, M, K, K2, N, Features, SeqLen, Hidden, Classes, Layer, H);

type B = Flex<f32>;

fn dev() -> burn::prelude::Device<B> {
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
    assert_eq!(c.shape().to_vec(), [3, 5]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn from_data_and_from_floats() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> = NamedTensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
        &dev,
    );
    let b: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::from_floats(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2], &dev);
    assert_eq!(a.dim_names(), &["M", "N"]);
    assert_eq!(b.dim_names(), &["M", "N"]);
    assert_eq!(a.shape().to_vec(), [2, 2]);
    a.inner.into_data().assert_eq(&b.inner.into_data(), true);
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
    assert_eq!(c.shape().to_vec(), [3, 5]);
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
    assert_eq!(out.shape().to_vec(), [3, 5]);
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
    assert_eq!(out.shape().to_vec(), [3, 5]);
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
    assert_eq!(out.shape().to_vec(), [5, 3]);
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
    let c: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs);
    assert_eq!(c.dim_names(), &["M", "N"]);
    assert_eq!(c.shape().to_vec(), [3, 5]);
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
    let c: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs);
    assert_eq!(c.dim_names(), &["M", "N"]);
    assert_eq!(c.shape().to_vec(), [3, 5]);
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
    let out: NamedTensor<B, dims![Batch, M, N], 3> = matmul(lhs, rhs);
    assert_eq!(out.dim_names(), &["Batch", "M", "N"]);
    assert_eq!(out.shape().to_vec(), [2, 3, 5]);
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
    let out: NamedTensor<B, dims![M, Batch, N], 3> = matmul(lhs, rhs);
    assert_eq!(out.dim_names(), &["M", "Batch", "N"]);
    assert_eq!(out.shape().to_vec(), [3, 2, 5]);
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
    let out: NamedTensor<B, dims![M, N, Batch], 3> = matmul(lhs, rhs);
    assert_eq!(out.dim_names(), &["M", "N", "Batch"]);
    assert_eq!(out.shape().to_vec(), [3, 5, 4]);
}

#[test]
fn matmul_multi_contract() {
    let dev = dev();
    // Contract over two dims (K and K2) simultaneously
    let lhs: NamedTensor<B, dims![M, K, K2], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
            [2usize, 3, 4],
        ),
        &dev,
    ));
    let rhs: NamedTensor<B, dims![K, K2, N], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=60).map(|x| x as f32 * 0.01).collect::<Vec<_>>(),
            [3usize, 4, 5],
        ),
        &dev,
    ));
    let out: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().to_vec(), [2, 5]);
}

#[test]
fn matmul_to_scalar() {
    let dev = dev();
    let lhs: NamedTensor<B, dims![K], 1> =
        NamedTensor::new(Tensor::from_data([1.0f32, 2.0, 3.0], &dev));
    let rhs: NamedTensor<B, dims![K], 1> =
        NamedTensor::new(Tensor::from_data([4.0f32, 5.0, 6.0], &dev));
    let s: f32 = matmul(lhs, rhs);
    assert!((s - 32.0).abs() < 1e-4, "expected 32.0, got {s}");
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
    assert_eq!(out.shape().to_vec(), [3]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 20.0).abs() < 1e-4, "expected mean 20.0, got {mean}");
}

#[test]
fn dot_partial_contraction() {
    // lhs: (Batch=2, Features=3), rhs: (Features=3, Classes=4)
    // shared: Features → contracted; output: (Batch=2, Classes=4)
    let dev = dev();
    let lhs: NamedTensor<B, dims![Batch, Features], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], // 2×3 identity-ish
            [2usize, 3],
        ),
        &dev,
    ));
    // rhs is 3×4 where each column j is filled with (j+1)
    let rhs: NamedTensor<B, dims![Features, Classes], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            vec![
                1.0f32, 2.0, 3.0, 4.0, // row 0
                1.0, 2.0, 3.0, 4.0, // row 1
                1.0, 2.0, 3.0, 4.0, // row 2
            ],
            [3usize, 4],
        ),
        &dev,
    ));
    let out: NamedTensor<B, dims![Batch, Classes], 2> = dot(lhs, rhs);
    assert_eq!(out.dim_names(), &["Batch", "Classes"]);
    assert_eq!(out.shape().to_vec(), [2, 4]);
    // row 0 of lhs is [1,0,0], dot with each rhs column → [1,2,3,4]
    // row 1 of lhs is [0,1,0], dot with each rhs column → [1,2,3,4]
    let data: Vec<f32> = out.inner.to_data().to_vec().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn dot_partial_contraction_reversed_output() {
    // Same as above but output dims in reverse order: (Classes, Batch)
    let dev = dev();
    let lhs: NamedTensor<B, dims![Batch, Features], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    let rhs: NamedTensor<B, dims![Features, Classes], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0], [3usize, 2]),
        &dev,
    ));
    // lhs×rhs = [[1,2],[4,5]] in (Batch, Classes) order
    // Transposed to (Classes, Batch): [[1,4],[2,5]]
    let out: NamedTensor<B, dims![Classes, Batch], 2> = dot(lhs, rhs);
    assert_eq!(out.dim_names(), &["Classes", "Batch"]);
    assert_eq!(out.shape().to_vec(), [2, 2]);
    let data: Vec<f32> = out.inner.to_data().to_vec().unwrap();
    assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0]);
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
    assert_eq!(t.shape().to_vec(), [2, 3, 5]);
    let t2: NamedTensor<B, dims![N, Batch, M], 3> = permute(t);
    assert_eq!(t2.dim_names(), &["N", "Batch", "M"]);
    assert_eq!(t2.shape().to_vec(), [5, 2, 3]);
}

#[test]
fn sum_and_rename() {
    let dev = dev();
    let t: NamedTensor<B, dims![SeqLen, Features], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([4usize, 8]), &dev));
    let s: NamedTensor<B, dims![Features], 1> = sum::<B, dims![SeqLen], _, _, _, 2, 1>(t);
    assert_eq!(s.dim_names(), &["Features"]);
    assert_eq!(s.shape().to_vec(), [8]);
    let h: NamedTensor<B, dims![Hidden], 1> = rename::<B, Features, Hidden, _, _, _, 1>(s);
    assert_eq!(h.dim_names(), &["Hidden"]);
}

#[test]
fn sum_to_scalar() {
    let dev = dev();
    let t: NamedTensor<B, dims![SeqLen, Features], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([4usize, 8]), &dev));
    let s: NamedTensor<B, dims![Features], 1> = sum::<B, dims![SeqLen], _, _, _, 2, 1>(t);
    let h: NamedTensor<B, dims![Hidden], 1> = rename::<B, Features, Hidden, _, _, _, 1>(s);
    let total: f32 = sum::<B, dims![Hidden], _, _, _, 1, 0>(h);
    assert!((total - 32.0).abs() < 1e-4, "expected 32.0, got {total}");
}

#[test]
fn mean_reduce() {
    let dev = dev();
    let t: NamedTensor<B, dims![SeqLen, Features], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2usize, 4]),
        &dev,
    ));
    let m: NamedTensor<B, dims![Features], 1> = t.mean::<dims![SeqLen], _, _, 1>();
    assert_eq!(m.dim_names(), &["Features"]);
    assert_eq!(m.shape().to_vec(), [4]);
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
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2usize, 4]),
        &dev,
    ));
    let s: f32 = t.mean::<dims![SeqLen, Features], _, _, 0>();
    assert!((s - 4.5).abs() < 1e-4, "expected 4.5, got {s}");
}

#[test]
fn mean_multi_dim_partial() {
    let dev = dev();
    let t: NamedTensor<B, dims![Batch, SeqLen, Features], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
            [2usize, 3, 4],
        ),
        &dev,
    ));
    let m: NamedTensor<B, dims![Features], 1> = t.mean::<dims![Batch, SeqLen], _, _, 1>();
    assert_eq!(m.dim_names(), &["Features"]);
    assert_eq!(m.shape().to_vec(), [4]);
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
    assert_eq!(out.shape().to_vec(), [3, 5]);
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
    assert_eq!(out.shape().to_vec(), [3, 5]);
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
    assert_eq!(out.shape().to_vec(), [3, 5]);
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
    assert_eq!(back.shape().to_vec(), [3, 5]);
    let mean: f32 = back.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4);
}

#[test]
fn untyped_roundtrip_permuted() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));

    let u = a.untyped();
    let back: NamedTensor<B, dims![N, M], 2> = u.to_named();
    assert_eq!(back.dim_names(), &["N", "M"]);
    assert_eq!(back.shape().to_vec(), [3, 2]);
}

fn arange_mn(dev: &burn::prelude::Device<B>) -> NamedTensor<B, dims![M, N], 2> {
    NamedTensor::new(Tensor::from_data(
        TensorData::new((0..24).map(|x| x as f32).collect::<Vec<_>>(), [4usize, 6]),
        dev,
    ))
}

#[test]
fn slice_resolves_dims_by_name_in_any_order() {
    let t = arange_mn(&dev());
    let out = t.slice(s![N => 1..3, M => 2..4]);
    assert_eq!(out.shape().to_vec(), [2, 2]);
    out.inner
        .into_data()
        .assert_eq(&TensorData::from([[13.0f32, 14.0], [19.0, 20.0]]), true);
}

#[test]
fn slice_keeps_unmentioned_dims_whole() {
    let t = arange_mn(&dev());
    let out = t.slice(s![N => 0..2]);
    assert_eq!(out.shape().to_vec(), [4, 2]);
    assert_eq!(out.dim_names(), &["M", "N"]);
}

#[test]
fn slice_supports_steps_and_negative_indices() {
    let t = arange_mn(&dev());
    let out = t.slice(s![N => 0..6;2, M => -1..]);
    assert_eq!(out.shape().to_vec(), [1, 3]);
    out.inner
        .into_data()
        .assert_eq(&TensorData::from([[18.0f32, 20.0, 22.0]]), true);
}

#[test]
fn slice_spec_is_reusable_across_layouts() {
    let a = arange_mn(&dev());
    let b: NamedTensor<B, dims![N, M], 2> = permute(a.clone());

    let spec = s![M => 0..2, N => 0..3];
    let sa = a.slice(spec.clone());
    let sb = b.slice(spec);

    // Positions resolved per-tensor: M is axis 0 in `a` but axis 1 in `b`.
    assert_eq!(sa.shape().to_vec(), [2, 3]);
    assert_eq!(sb.shape().to_vec(), [3, 2]);
    let sb_mn: NamedTensor<B, dims![M, N], 2> = permute(sb);
    sb_mn
        .inner
        .into_data()
        .assert_eq(&sa.inner.into_data(), true);
}

#[test]
fn slice_by_slices_a_single_named_dim() {
    let t = arange_mn(&dev());
    let out = t.slice_by(N, s![1..4]);
    assert_eq!(out.shape().to_vec(), [4, 3]);
}

#[test]
fn slice_assign_writes_the_named_region() {
    let dev = dev();
    let t = arange_mn(&dev);
    let values: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::zeros(Shape::new([2usize, 2]), &dev));
    let out = t.slice_assign(s![N => 1..3, M => 2..4], values);
    out.inner.into_data().assert_eq(
        &TensorData::from([
            [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [12.0, 0.0, 0.0, 15.0, 16.0, 17.0],
            [18.0, 0.0, 0.0, 21.0, 22.0, 23.0],
        ]),
        true,
    );
}

#[test]
fn slice_fill_fills_the_named_region() {
    let t = arange_mn(&dev());
    let out = t.slice_fill(s![M => 0..1], -1.0);
    out.inner.into_data().assert_eq(
        &TensorData::from([
            [-1.0f32, -1.0, -1.0, -1.0, -1.0, -1.0],
            [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
        ]),
        true,
    );
}

#[test]
fn isel_by_drops_the_named_dim() {
    let dev = dev();
    let t: NamedTensor<B, dims![Batch, SeqLen, Hidden], 3> = NamedTensor::new(Tensor::from_data(
        TensorData::new(
            (0..24).map(|x| x as f32).collect::<Vec<_>>(),
            [2usize, 3, 4],
        ),
        &dev,
    ));

    let out: NamedTensor<B, dims![Batch, Hidden], 2> = t.clone().isel_by(SeqLen, 1);
    assert_eq!(out.dim_names(), &["Batch", "Hidden"]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[4.0f32, 5.0, 6.0, 7.0], [16.0, 17.0, 18.0, 19.0]]),
        true,
    );

    // Negative indices count from the end.
    let last: NamedTensor<B, dims![Batch, SeqLen], 2> = t.isel_by(Hidden, -1);
    last.inner.into_data().assert_eq(
        &TensorData::from([[3.0f32, 7.0, 11.0], [15.0, 19.0, 23.0]]),
        true,
    );
}

#[test]
fn concat_along_named_dim() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
        &dev,
    ));
    let b: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], [2usize, 2]),
        &dev,
    ));
    // concat along M (axis 0): rows stack
    let out: NamedTensor<B, dims![M, N], 2> = concat(vec![a, b], M);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().to_vec(), [4, 2]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        true,
    );
}

#[test]
fn concat_along_second_named_dim() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
        &dev,
    ));
    let b: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], [2usize, 2]),
        &dev,
    ));
    // concat along N (axis 1): columns stack
    let out: NamedTensor<B, dims![M, N], 2> = concat(vec![a, b], N);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().to_vec(), [2, 4]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[1.0f32, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]),
        true,
    );
}

#[test]
fn stack_prepends_a_new_named_dim() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
        &dev,
    ));
    let b: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], [2usize, 2]),
        &dev,
    ));
    // stack along a new dim `Layer` → dims![Layer, M, N]
    let out: NamedTensor<B, dims![Layer, M, N], 3> = stack::<B, _, Layer, 2, 3>(vec![a, b], Layer);
    assert_eq!(out.dim_names(), &["Layer", "M", "N"]);
    assert_eq!(out.shape().to_vec(), [2, 2, 2]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        true,
    );
}

#[test]
fn stack_then_isel_roundtrips() {
    let dev = dev();
    let a: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
        &dev,
    ));
    let b: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], [2usize, 2]),
        &dev,
    ));
    let stacked: NamedTensor<B, dims![Layer, M, N], 3> =
        stack::<B, _, Layer, 2, 3>(vec![a.clone(), b], Layer);
    // isel_by(Layer, 0) recovers the first input
    let first: NamedTensor<B, dims![M, N], 2> = stacked.isel_by(Layer, 0);
    first
        .inner
        .into_data()
        .assert_eq(&a.inner.into_data(), true);
}

#[test]
fn squeeze_dim_removes_a_unit_dim() {
    let dev = dev();
    let t: NamedTensor<B, dims![Batch, M, N], 3> =
        NamedTensor::new(Tensor::ones(Shape::new([1usize, 3, 5]), &dev));
    let out: NamedTensor<B, dims![M, N], 2> = t.squeeze_dim(Batch);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
}

#[test]
#[should_panic(expected = "size is not 1")]
fn squeeze_dim_panics_on_non_unit_dim() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 5]), &dev));
    let _: NamedTensor<B, dims![N], 1> = t.squeeze_dim(M);
}

#[test]
fn squeeze_removes_the_listed_dims() {
    let dev = dev();
    let t: NamedTensor<B, dims![Batch, M, K, N], 4> =
        NamedTensor::new(Tensor::ones(Shape::new([1usize, 3, 1, 5]), &dev));
    let out: NamedTensor<B, dims![M, N], 2> = t.squeeze::<dims![Batch, K], _, _, 2>();
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
}

// ── align_to / align_as ──

#[test]
fn align_to_adds_size1_dims() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    // Insert a new dim `K` in the middle: dims![M, K, N]
    let out: NamedTensor<B, dims![M, K, N], 3> = align_to(t);
    assert_eq!(out.dim_names(), &["M", "K", "N"]);
    assert_eq!(out.shape().to_vec(), [2, 1, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 1, 3]),
        true,
    );
}

#[test]
fn align_to_permutes() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    // Reorder to dims![N, M]
    let out: NamedTensor<B, dims![N, M], 2> = align_to(t);
    assert_eq!(out.dim_names(), &["N", "M"]);
    assert_eq!(out.shape().to_vec(), [3, 2]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], [3usize, 2]),
        true,
    );
}

#[test]
fn align_to_method_form() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    let out: NamedTensor<B, dims![N, K, M], 3> = t.align_to();
    assert_eq!(out.dim_names(), &["N", "K", "M"]);
    assert_eq!(out.shape().to_vec(), [3, 1, 2]);
}

#[test]
fn align_as_matches_other() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    let other: NamedTensor<B, dims![N, K, M], 3> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 1, 2]), &dev));
    let out: NamedTensor<B, dims![N, K, M], 3> = align_as(t, &other);
    assert_eq!(out.dim_names(), &["N", "K", "M"]);
    assert_eq!(out.shape().to_vec(), [3, 1, 2]);
}

#[test]
fn align_as_method_form() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    let other: NamedTensor<B, dims![N, K, M], 3> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 1, 2]), &dev));
    let out: NamedTensor<B, dims![N, K, M], 3> = t.align_as(&other);
    assert_eq!(out.dim_names(), &["N", "K", "M"]);
    assert_eq!(out.shape().to_vec(), [3, 1, 2]);
}

#[test]
fn align_to_identity_is_noop() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    // Target equals source: no permute, no insert.
    let out: NamedTensor<B, dims![M, N], 2> = align_to(t);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().to_vec(), [2, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        true,
    );
}

#[test]
fn align_to_prepends_new_dim() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    // New dim `K` at the front.
    let out: NamedTensor<B, dims![K, M, N], 3> = align_to(t);
    assert_eq!(out.dim_names(), &["K", "M", "N"]);
    assert_eq!(out.shape().to_vec(), [1, 2, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1usize, 2, 3]),
        true,
    );
}

#[test]
fn align_to_adds_multiple_dims() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    // Two new dims, one at the front and one in the middle.
    let out: NamedTensor<B, dims![K, M, H, N], 4> = align_to(t);
    assert_eq!(out.dim_names(), &["K", "M", "H", "N"]);
    assert_eq!(out.shape().to_vec(), [1, 2, 1, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1usize, 2, 1, 3]),
        true,
    );
}

#[test]
fn align_as_verifies_data() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    let other: NamedTensor<B, dims![N, K, M], 3> =
        NamedTensor::new(Tensor::ones(Shape::new([3usize, 1, 2]), &dev));
    // Combined permute (M,N → N,M) + insert (K): flat data is the transpose.
    let out: NamedTensor<B, dims![N, K, M], 3> = align_as(t, &other);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], [3usize, 1, 2]),
        true,
    );
}

#[test]
fn align_to_rank1() {
    let dev = dev();
    let t: NamedTensor<B, dims![M], 1> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0], [3usize]),
        &dev,
    ));
    let out: NamedTensor<B, dims![K, M], 2> = align_to(t);
    assert_eq!(out.dim_names(), &["K", "M"]);
    assert_eq!(out.shape().to_vec(), [1, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0], [1usize, 3]),
        true,
    );
}

#[test]
fn cumsum_along_named_dim() {
    let dev = dev();
    let t: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(Tensor::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        &dev,
    ));
    // cumsum along N (axis 1): running total within each row
    let out = t.cumsum(N);
    assert_eq!(out.dim_names(), &["M", "N"]);
    assert_eq!(out.shape().to_vec(), [2, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[1.0f32, 3.0, 6.0], [4.0, 9.0, 15.0]]),
        true,
    );
}
