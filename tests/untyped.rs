use burn::backend::Flex;
use burn::tensor::{Shape, Tensor, TensorData};
use named_tensor::{self as untyped, NamedTensor};

type B = Flex<f32>;

fn dev() -> burn::prelude::Device<B> {
    Default::default()
}

#[test]
fn add_same_shape() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(["M", "N"], Tensor::ones(Shape::new([3usize, 5]), &dev));
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c: NamedTensor<B, 2> = untyped::add(a, b);
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(c.shape().to_vec(), [3, 5]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn from_data_and_from_floats() {
    let dev = dev();
    let a: NamedTensor<B, 2> = NamedTensor::from_data(
        ["M", "N"],
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
        &dev,
    );
    let b: NamedTensor<B, 2> =
        NamedTensor::from_floats(["M", "N"], vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2], &dev);
    assert_eq!(a.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(b.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(a.shape().to_vec(), [2, 2]);
    a.inner.into_data().assert_eq(&b.inner.into_data(), true);
}

#[test]
fn add_with_plus_operator() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(["M", "N"], Tensor::ones(Shape::new([3usize, 5]), &dev));
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c = a + b;
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(c.shape().to_vec(), [3, 5]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn add_rank2_rank1_broadcast() {
    let dev = dev();
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new((1..=15).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 5]),
            &dev,
        ),
    );
    let bias =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([0.1f32, 0.2, 0.3, 0.4, 0.5], &dev));
    let out: NamedTensor<B, 2> = untyped::add(mat, bias);
    assert_eq!(out.names(), &["N".to_string(), "M".to_string()]);
    assert_eq!(out.shape().to_vec(), [5, 3]);
}

#[test]
fn add_disjoint_dims() {
    let dev = dev();
    let row = NamedTensor::<B, 1>::new(["M"], Tensor::from_data([1.0f32, 2.0, 3.0], &dev));
    let col = NamedTensor::<B, 1>::new(
        ["N"],
        Tensor::from_data([10.0f32, 20.0, 30.0, 40.0, 50.0], &dev),
    );
    let out: NamedTensor<B, 2> = untyped::add(row, col);
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
}

#[test]
fn add_commuted_order() {
    let dev = dev();
    let bias =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([1.0f32, 1.0, 1.0, 1.0, 1.0], &dev));
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let out: NamedTensor<B, 2> = untyped::add(bias, mat);
    assert_eq!(out.names(), &["N".to_string(), "M".to_string()]);
    assert_eq!(out.shape().to_vec(), [5, 3]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn matmul_2d_standard() {
    let dev = dev();
    let lhs_data: Vec<f32> = (0..3).flat_map(|r| vec![(r + 1) as f32; 4]).collect();
    let lhs = NamedTensor::<B, 2>::new(
        ["M", "K"],
        Tensor::from_data(TensorData::new(lhs_data, [3usize, 4]), &dev),
    );
    let rhs_data: Vec<f32> = (0..4)
        .flat_map(|_| (1..=5).map(|c| c as f32 * 0.1).collect::<Vec<_>>())
        .collect();
    let rhs = NamedTensor::<B, 2>::new(
        ["K", "N"],
        Tensor::from_data(TensorData::new(rhs_data, [4usize, 5]), &dev),
    );
    let c: NamedTensor<B, 2> = untyped::matmul(lhs, rhs, "K");
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(c.shape().to_vec(), [3, 5]);
}

#[test]
fn matmul_2d_k_nonstandard() {
    let dev = dev();
    let lhs = NamedTensor::<B, 2>::new(
        ["K", "M"],
        Tensor::from_data(
            TensorData::new((1..=12).map(|x| x as f32).collect::<Vec<_>>(), [4usize, 3]),
            &dev,
        ),
    );
    let rhs = NamedTensor::<B, 2>::new(
        ["N", "K"],
        Tensor::from_data(
            TensorData::new(
                (1..=20).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                [5usize, 4],
            ),
            &dev,
        ),
    );
    let c: NamedTensor<B, 2> = untyped::matmul(lhs, rhs, "K");
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(c.shape().to_vec(), [3, 5]);
}

#[test]
fn matmul_3d_batched() {
    let dev = dev();
    let lhs = NamedTensor::<B, 3>::new(
        ["Batch", "M", "K"],
        Tensor::from_data(
            TensorData::new(
                (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                [2usize, 3, 4],
            ),
            &dev,
        ),
    );
    let rhs = NamedTensor::<B, 3>::new(
        ["Batch", "K", "N"],
        Tensor::from_data(
            TensorData::new(
                (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                [2usize, 4, 5],
            ),
            &dev,
        ),
    );
    let out: NamedTensor<B, 3> = untyped::matmul(lhs, rhs, "K");
    assert_eq!(
        out.names(),
        &["Batch".to_string(), "M".to_string(), "N".to_string()]
    );
    assert_eq!(out.shape().to_vec(), [2, 3, 5]);
}

#[test]
fn matmul_3d_k_middle() {
    let dev = dev();
    let lhs = NamedTensor::<B, 3>::new(
        ["M", "K", "Batch"],
        Tensor::from_data(
            TensorData::new(
                (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                [3usize, 4, 2],
            ),
            &dev,
        ),
    );
    let rhs = NamedTensor::<B, 3>::new(
        ["Batch", "K", "N"],
        Tensor::from_data(
            TensorData::new(
                (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                [2usize, 4, 5],
            ),
            &dev,
        ),
    );
    let out: NamedTensor<B, 3> = untyped::matmul(lhs, rhs, "K");
    assert_eq!(
        out.names(),
        &["Batch".to_string(), "M".to_string(), "N".to_string()]
    );
    assert_eq!(out.shape().to_vec(), [2, 3, 5]);
}

#[test]
fn matmul_mixed_rank() {
    let dev = dev();
    let lhs = NamedTensor::<B, 2>::new(
        ["M", "K"],
        Tensor::from_data(
            TensorData::new((1..=6).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 2]),
            &dev,
        ),
    );
    let rhs = NamedTensor::<B, 3>::new(
        ["K", "N", "Batch"],
        Tensor::from_data(
            TensorData::new(
                (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                [2usize, 5, 4],
            ),
            &dev,
        ),
    );
    let out: NamedTensor<B, 3> = untyped::matmul(lhs, rhs, "K");
    assert_eq!(
        out.names(),
        &["M".to_string(), "N".to_string(), "Batch".to_string()]
    );
    assert_eq!(out.shape().to_vec(), [3, 5, 4]);
}

#[test]
fn matmul_double_contract() {
    let dev = dev();
    let lhs = NamedTensor::<B, 3>::new(
        ["A", "K1", "K2"],
        Tensor::from_data(
            TensorData::new(
                (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                [4usize, 2, 3],
            ),
            &dev,
        ),
    );
    let rhs = NamedTensor::<B, 3>::new(
        ["K2", "K1", "B"],
        Tensor::from_data(
            TensorData::new(
                (1..=30).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                [3usize, 2, 5],
            ),
            &dev,
        ),
    );
    let out: NamedTensor<B, 2> = untyped::matmul(lhs, rhs, ["K1", "K2"]);
    assert_eq!(out.names(), &["A".to_string(), "B".to_string()]);
    assert_eq!(out.shape().to_vec(), [4, 5]);
}

#[test]
fn dot_product() {
    let dev = dev();
    let u = NamedTensor::<B, 1>::new(
        ["Features"],
        Tensor::from_data([1.0f32, 2.0, 3.0, 4.0], &dev),
    );
    let v = NamedTensor::<B, 1>::new(
        ["Features"],
        Tensor::from_data([0.25f32, 0.5, 0.75, 1.0], &dev),
    );
    let result = untyped::dot(u, v);
    assert!((result - 7.5).abs() < 1e-4, "expected 7.5, got {result}");
}

#[test]
fn permute() {
    let dev = dev();
    let t = NamedTensor::<B, 3>::new(
        ["Batch", "M", "N"],
        Tensor::from_data(
            TensorData::new(
                (1..=30).map(|x| x as f32).collect::<Vec<_>>(),
                [2usize, 3, 5],
            ),
            &dev,
        ),
    );
    assert_eq!(t.shape().to_vec(), [2, 3, 5]);
    let t2: NamedTensor<B, 3> = untyped::permute(t, ["N", "Batch", "M"]);
    assert_eq!(
        t2.names(),
        &["N".to_string(), "Batch".to_string(), "M".to_string()]
    );
    assert_eq!(t2.shape().to_vec(), [5, 2, 3]);
}

#[test]
fn sum_and_rename() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["SeqLen", "Features"],
        Tensor::ones(Shape::new([4usize, 8]), &dev),
    );
    let s: NamedTensor<B, 1> = untyped::sum(t, "SeqLen");
    assert_eq!(s.names(), &["Features".to_string()]);
    assert_eq!(s.shape().to_vec(), [8]);
    let h: NamedTensor<B, 1> = untyped::rename(s, "Features", "Hidden");
    assert_eq!(h.names(), &["Hidden".to_string()]);
}

#[test]
fn add_operator_broadcast() {
    let dev = dev();
    let mat = NamedTensor::<B, 2>::new(["M", "N"], Tensor::ones(Shape::new([3usize, 5]), &dev));
    let bias =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0], &dev));
    let out = mat + bias;
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
}

// ── sub ──

#[test]
fn sub_same_shape() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 5.0,
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c: NamedTensor<B, 2> = untyped::sub(a, b);
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn sub_with_minus_operator() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 5.0,
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c = a - b;
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn sub_broadcast() {
    let dev = dev();
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 10.0,
    );
    let bias =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0], &dev));
    let out: NamedTensor<B, 2> = untyped::sub(mat, bias);
    assert_eq!(out.shape().to_vec(), [5, 3]);
}

#[test]
fn sub_operator_broadcast() {
    let dev = dev();
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 10.0,
    );
    let bias =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0], &dev));
    let out = mat - bias;
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
}

// ── mul ──

#[test]
fn mul_same_shape() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 3.0,
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c: NamedTensor<B, 2> = untyped::mul(a, b);
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 6.0).abs() < 1e-4, "expected mean 6.0, got {mean}");
}

#[test]
fn mul_with_star_operator() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 3.0,
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c = a * b;
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 6.0).abs() < 1e-4, "expected mean 6.0, got {mean}");
}

#[test]
fn mul_broadcast() {
    let dev = dev();
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 4.0,
    );
    let scale =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([2.0f32, 2.0, 2.0, 2.0, 2.0], &dev));
    let out: NamedTensor<B, 2> = untyped::mul(mat, scale);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 8.0).abs() < 1e-4, "expected mean 8.0, got {mean}");
}

#[test]
fn mul_operator_broadcast() {
    let dev = dev();
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 4.0,
    );
    let scale =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([2.0f32, 2.0, 2.0, 2.0, 2.0], &dev));
    let out = mat * scale;
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 8.0).abs() < 1e-4, "expected mean 8.0, got {mean}");
}

// ── div ──

#[test]
fn div_same_shape() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 6.0,
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c: NamedTensor<B, 2> = untyped::div(a, b);
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn div_with_slash_operator() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 6.0,
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c = a / b;
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    let mean: f32 = c.inner.mean().into_scalar();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn div_broadcast() {
    let dev = dev();
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 10.0,
    );
    let scale =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([2.0f32, 2.0, 2.0, 2.0, 2.0], &dev));
    let out: NamedTensor<B, 2> = untyped::div(mat, scale);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 5.0).abs() < 1e-4, "expected mean 5.0, got {mean}");
}

#[test]
fn mean_reduce() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["SeqLen", "Features"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2usize, 4]),
            &dev,
        ),
    );
    let m: NamedTensor<B, 1> = t.mean(["SeqLen"]);
    assert_eq!(m.names(), &["Features".to_string()]);
    assert_eq!(m.shape().to_vec(), [4]);
    // mean over SeqLen: [(1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2] = [3, 4, 5, 6]
    let val: f32 = m.inner.mean().into_scalar();
    assert!((val - 4.5).abs() < 1e-4, "expected 4.5, got {val}");
}

#[test]
fn mean_multi_dim() {
    let dev = dev();
    let t = NamedTensor::<B, 3>::new(
        ["Batch", "SeqLen", "Features"],
        Tensor::from_data(
            TensorData::new(
                (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                [2usize, 3, 4],
            ),
            &dev,
        ),
    );
    let m: NamedTensor<B, 1> = t.mean(["Batch", "SeqLen"]);
    assert_eq!(m.names(), &["Features".to_string()]);
    assert_eq!(m.shape().to_vec(), [4]);
    // mean over Batch and SeqLen for each feature
    let val: f32 = m.inner.mean().into_scalar();
    assert!((val - 12.5).abs() < 1e-4, "expected 12.5, got {val}");
}

#[test]
fn mean_multi_dim_partial() {
    let dev = dev();
    let t = NamedTensor::<B, 3>::new(
        ["Batch", "SeqLen", "Features"],
        Tensor::from_data(
            TensorData::new(
                (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                [2usize, 3, 4],
            ),
            &dev,
        ),
    );
    let m: NamedTensor<B, 1> = t.mean(["Batch", "SeqLen"]);
    assert_eq!(m.names(), &["Features".to_string()]);
    assert_eq!(m.shape().to_vec(), [4]);
    let val: f32 = m.inner.mean().into_scalar();
    assert!((val - 12.5).abs() < 1e-4, "expected 12.5, got {val}");
}

#[test]
fn div_operator_broadcast() {
    let dev = dev();
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 10.0,
    );
    let scale =
        NamedTensor::<B, 1>::new(["N"], Tensor::from_data([2.0f32, 2.0, 2.0, 2.0, 2.0], &dev));
    let out = mat / scale;
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
    let mean: f32 = out.inner.mean().into_scalar();
    assert!((mean - 5.0).abs() < 1e-4, "expected mean 5.0, got {mean}");
}

fn arange_mn(dev: &burn::prelude::Device<B>) -> NamedTensor<B, 2> {
    NamedTensor::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new((0..24).map(|x| x as f32).collect::<Vec<_>>(), [4usize, 6]),
            dev,
        ),
    )
}

#[test]
fn slice_with_string_keys() {
    let t = arange_mn(&dev());
    let out = t.slice(untyped::s!["N" => 1..3, "M" => 2..4]);
    assert_eq!(out.shape().to_vec(), [2, 2]);
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    out.inner
        .into_data()
        .assert_eq(&TensorData::from([[13.0f32, 14.0], [19.0, 20.0]]), true);
}

#[test]
fn slice_keeps_unmentioned_dims_and_supports_steps() {
    let t = arange_mn(&dev());
    let out = t.slice(untyped::s!["N" => 0..6;2]);
    assert_eq!(out.shape().to_vec(), [4, 3]);
}

#[test]
fn slice_by_single_dim() {
    let t = arange_mn(&dev());
    let out = t.slice_by("N", untyped::s![1..4]);
    assert_eq!(out.shape().to_vec(), [4, 3]);
}

#[test]
fn slice_assign_aligns_values_by_name() {
    let dev = dev();
    let t = arange_mn(&dev);
    // Values arrive with axes in the opposite order; alignment is by name.
    let values = NamedTensor::<B, 2>::new(
        ["N", "M"],
        Tensor::zeros(burn::tensor::Shape::new([2usize, 2]), &dev),
    );
    let out = t.slice_assign(untyped::s!["N" => 1..3, "M" => 2..4], values);
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
    let out = t.slice_fill(untyped::s!["M" => 0..1], -1.0);
    let first_row: Vec<f32> = out
        .inner
        .into_data()
        .to_vec()
        .unwrap()
        .into_iter()
        .take(6)
        .collect();
    assert_eq!(first_row, vec![-1.0f32; 6]);
}

#[test]
fn isel_by_drops_the_named_dim() {
    let t = arange_mn(&dev());
    let out: NamedTensor<B, 1> = t.isel_by("M", -1);
    assert_eq!(out.names(), &["N".to_string()]);
    out.inner.into_data().assert_eq(
        &TensorData::from([18.0f32, 19.0, 20.0, 21.0, 22.0, 23.0]),
        true,
    );
}

#[test]
#[should_panic(expected = "not found")]
fn slice_with_unknown_dim_panics() {
    let t = arange_mn(&dev());
    let _ = t.slice(untyped::s!["Z" => 0..1]);
}

#[test]
fn concat_along_named_dim() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
            &dev,
        ),
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], [2usize, 2]),
            &dev,
        ),
    );
    let out: NamedTensor<B, 2> = untyped::concat(vec![a, b], "M");
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [4, 2]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        true,
    );
}

#[test]
fn concat_aligns_differing_axis_orders() {
    let dev = dev();
    // `a` is (M, N), `b` is (N, M) — concat aligns by name first.
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
            &dev,
        ),
    );
    let b = NamedTensor::<B, 2>::new(
        ["N", "M"],
        Tensor::from_data(
            TensorData::new(vec![5.0f32, 7.0, 6.0, 8.0], [2usize, 2]),
            &dev,
        ),
    );
    let out: NamedTensor<B, 2> = untyped::concat(vec![a, b], "M");
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [4, 2]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        true,
    );
}

#[test]
fn stack_prepends_a_new_named_dim() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2usize, 2]),
            &dev,
        ),
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], [2usize, 2]),
            &dev,
        ),
    );
    let out: NamedTensor<B, 3> = untyped::stack::<B, 2, 3>(vec![a, b], "Layer");
    assert_eq!(
        out.names(),
        &["Layer".to_string(), "M".to_string(), "N".to_string()]
    );
    assert_eq!(out.shape().to_vec(), [2, 2, 2]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        true,
    );
}

#[test]
fn squeeze_dim_removes_a_unit_dim() {
    let dev = dev();
    let t = NamedTensor::<B, 3>::new(
        ["Batch", "M", "N"],
        Tensor::ones(Shape::new([1usize, 3, 5]), &dev),
    );
    let out: NamedTensor<B, 2> = t.squeeze_dim("Batch");
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
}

#[test]
fn squeeze_removes_all_unit_dims() {
    let dev = dev();
    let t = NamedTensor::<B, 4>::new(
        ["Batch", "M", "K", "N"],
        Tensor::ones(Shape::new([1usize, 3, 1, 5]), &dev),
    );
    let out: NamedTensor<B, 2> = t.squeeze();
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [3, 5]);
}

#[test]
#[should_panic(expected = "expected D_OUT")]
fn squeeze_panics_on_rank_mismatch() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(["M", "N"], Tensor::ones(Shape::new([3usize, 5]), &dev));
    // No unit dims, so squeezing to rank 1 is a runtime error.
    let _: NamedTensor<B, 1> = t.squeeze();
}

// ── align_to / align_as ──

#[test]
fn align_to_adds_size1_dims() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    let out: NamedTensor<B, 3> = untyped::align_to(t, ["M", "K", "N"]);
    assert_eq!(
        out.names(),
        &["M".to_string(), "K".to_string(), "N".to_string()]
    );
    assert_eq!(out.shape().to_vec(), [2, 1, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 1, 3]),
        true,
    );
}

#[test]
fn align_to_permutes() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    let out: NamedTensor<B, 2> = untyped::align_to(t, ["N", "M"]);
    assert_eq!(out.names(), &["N".to_string(), "M".to_string()]);
    assert_eq!(out.shape().to_vec(), [3, 2]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], [3usize, 2]),
        true,
    );
}

#[test]
fn align_as_matches_other() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    let other = NamedTensor::<B, 3>::new(
        ["N", "K", "M"],
        Tensor::ones(Shape::new([3usize, 1, 2]), &dev),
    );
    let out: NamedTensor<B, 3> = untyped::align_as(t, &other);
    assert_eq!(
        out.names(),
        &["N".to_string(), "K".to_string(), "M".to_string()]
    );
    assert_eq!(out.shape().to_vec(), [3, 1, 2]);
}

#[test]
fn align_as_method_form() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    let other = NamedTensor::<B, 3>::new(
        ["N", "K", "M"],
        Tensor::ones(Shape::new([3usize, 1, 2]), &dev),
    );
    let out: NamedTensor<B, 3> = t.align_as(&other);
    assert_eq!(
        out.names(),
        &["N".to_string(), "K".to_string(), "M".to_string()]
    );
    assert_eq!(out.shape().to_vec(), [3, 1, 2]);
}

#[test]
#[should_panic(expected = "not in target")]
fn align_to_panics_on_missing_dim() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([2usize, 3]), &dev),
    );
    // `M` is missing from the target
    let _: NamedTensor<B, 2> = untyped::align_to(t, ["N", "K"]);
}

#[test]
fn align_to_identity_is_noop() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    let out: NamedTensor<B, 2> = untyped::align_to(t, ["M", "N"]);
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [2, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
        true,
    );
}

#[test]
fn align_to_prepends_new_dim() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    let out: NamedTensor<B, 3> = untyped::align_to(t, ["K", "M", "N"]);
    assert_eq!(
        out.names(),
        &["K".to_string(), "M".to_string(), "N".to_string()]
    );
    assert_eq!(out.shape().to_vec(), [1, 2, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1usize, 2, 3]),
        true,
    );
}

#[test]
fn align_to_adds_multiple_dims() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    let out: NamedTensor<B, 4> = untyped::align_to(t, ["K", "M", "H", "N"]);
    assert_eq!(
        out.names(),
        &[
            "K".to_string(),
            "M".to_string(),
            "H".to_string(),
            "N".to_string()
        ]
    );
    assert_eq!(out.shape().to_vec(), [1, 2, 1, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [1usize, 2, 1, 3]),
        true,
    );
}

#[test]
fn align_as_verifies_data() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    let other = NamedTensor::<B, 3>::new(
        ["N", "K", "M"],
        Tensor::ones(Shape::new([3usize, 1, 2]), &dev),
    );
    let out: NamedTensor<B, 3> = untyped::align_as(t, &other);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], [3usize, 1, 2]),
        true,
    );
}

#[test]
fn align_to_rank1() {
    let dev = dev();
    let t = NamedTensor::<B, 1>::new(
        ["M"],
        Tensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3usize]), &dev),
    );
    let out: NamedTensor<B, 2> = untyped::align_to(t, ["K", "M"]);
    assert_eq!(out.names(), &["K".to_string(), "M".to_string()]);
    assert_eq!(out.shape().to_vec(), [1, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::new(vec![1.0f32, 2.0, 3.0], [1usize, 3]),
        true,
    );
}

#[test]
fn cumsum_along_named_dim() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    // cumsum along N (axis 1): running total within each row
    let out = t.cumsum("N");
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [2, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[1.0f32, 3.0, 6.0], [4.0, 9.0, 15.0]]),
        true,
    );
}

#[test]
fn cumsum_prefix_sum() {
    let dev = dev();
    let t = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3]),
            &dev,
        ),
    );
    // prefix sum over both M and N: corner sums
    let out = t.cumsum(["M", "N"]);
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().to_vec(), [2, 3]);
    out.inner.into_data().assert_eq(
        &TensorData::from([[1.0f32, 3.0, 6.0], [5.0, 12.0, 21.0]]),
        true,
    );
}
