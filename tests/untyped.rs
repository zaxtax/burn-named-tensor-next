use burn::backend::NdArray;
use burn::tensor::{Shape, Tensor, TensorData};
use named_tensor::untyped::{self, NamedTensor};

type B = NdArray<f32>;

fn dev() -> <B as burn::prelude::Backend>::Device {
    Default::default()
}

#[test]
fn add_same_shape() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev),
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c: NamedTensor<B, 2> = untyped::add(a, b);
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(c.shape().dims, [3, 5]);
    let mean: f32 = c.inner.mean().into_scalar().into();
    assert!((mean - 3.0).abs() < 1e-4, "expected mean 3.0, got {mean}");
}

#[test]
fn add_with_plus_operator() {
    let dev = dev();
    let a = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev),
    );
    let b = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let c = a + b;
    assert_eq!(c.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(c.shape().dims, [3, 5]);
    let mean: f32 = c.inner.mean().into_scalar().into();
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
    let bias = NamedTensor::<B, 1>::new(
        ["N"],
        Tensor::from_data([0.1f32, 0.2, 0.3, 0.4, 0.5], &dev),
    );
    let out: NamedTensor<B, 2> = untyped::add(mat, bias);
    assert_eq!(out.names(), &["N".to_string(), "M".to_string()]);
    assert_eq!(out.shape().dims, [5, 3]);
}

#[test]
fn add_disjoint_dims() {
    let dev = dev();
    let row = NamedTensor::<B, 1>::new(
        ["M"],
        Tensor::from_data([1.0f32, 2.0, 3.0], &dev),
    );
    let col = NamedTensor::<B, 1>::new(
        ["N"],
        Tensor::from_data([10.0f32, 20.0, 30.0, 40.0, 50.0], &dev),
    );
    let out: NamedTensor<B, 2> = untyped::add(row, col);
    assert_eq!(out.names(), &["M".to_string(), "N".to_string()]);
    assert_eq!(out.shape().dims, [3, 5]);
}

#[test]
fn add_commuted_order() {
    let dev = dev();
    let bias = NamedTensor::<B, 1>::new(
        ["N"],
        Tensor::from_data([1.0f32, 1.0, 1.0, 1.0, 1.0], &dev),
    );
    let mat = NamedTensor::<B, 2>::new(
        ["M", "N"],
        Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
    );
    let out: NamedTensor<B, 2> = untyped::add(bias, mat);
    assert_eq!(out.names(), &["N".to_string(), "M".to_string()]);
    assert_eq!(out.shape().dims, [5, 3]);
    let mean: f32 = out.inner.mean().into_scalar().into();
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
    assert_eq!(c.shape().dims, [3, 5]);
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
    assert_eq!(c.shape().dims, [3, 5]);
}

#[test]
fn matmul_3d_batched() {
    let dev = dev();
    let lhs = NamedTensor::<B, 3>::new(
        ["Batch", "M", "K"],
        Tensor::from_data(
            TensorData::new((1..=24).map(|x| x as f32).collect::<Vec<_>>(), [2usize, 3, 4]),
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
    assert_eq!(out.shape().dims, [2, 3, 5]);
}

#[test]
fn matmul_3d_k_middle() {
    let dev = dev();
    let lhs = NamedTensor::<B, 3>::new(
        ["M", "K", "Batch"],
        Tensor::from_data(
            TensorData::new((1..=24).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 4, 2]),
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
    assert_eq!(out.shape().dims, [2, 3, 5]);
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
        &[
            "M".to_string(),
            "N".to_string(),
            "Batch".to_string()
        ]
    );
    assert_eq!(out.shape().dims, [3, 5, 4]);
}

#[test]
fn matmul_double_contract() {
    let dev = dev();
    let lhs = NamedTensor::<B, 3>::new(
        ["A", "K1", "K2"],
        Tensor::from_data(
            TensorData::new((1..=24).map(|x| x as f32).collect::<Vec<_>>(), [4usize, 2, 3]),
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
    assert_eq!(out.shape().dims, [4, 5]);
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
    assert_eq!(t.shape().dims, [2, 3, 5]);
    let t2: NamedTensor<B, 3> = untyped::permute(t, ["N", "Batch", "M"]);
    assert_eq!(
        t2.names(),
        &["N".to_string(), "Batch".to_string(), "M".to_string()]
    );
    assert_eq!(t2.shape().dims, [5, 2, 3]);
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
    assert_eq!(s.shape().dims, [8]);
    let h: NamedTensor<B, 1> = untyped::rename(s, "Features", "Hidden");
    assert_eq!(h.names(), &["Hidden".to_string()]);
}
