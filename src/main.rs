use named_tensor::{dim, dims, NamedTensor, add, matmul, dot, sum_dim, rename};
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData, Shape};

dim!(Batch, M, K, N, Features, SeqLen, Hidden);

type B = NdArray<f32>;

fn main() {
    let dev: <B as burn::prelude::Backend>::Device = Default::default();

    println!("=== named-tensor demo ===\n");

    // 1. add: same shape
    {
        let a: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(
            Tensor::ones(Shape::new([3usize, 5]), &dev));
        let b: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(
            Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
        let c: NamedTensor<B, dims![M, N], 2> = add(a, b);
        println!("1. add (M,N)+(M,N) → mean={:.1}\n",
            Into::<f32>::into(c.inner.mean().into_scalar()));
    }

    // 2. add: rank-2 + rank-1 broadcast
    {
        let mat: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(
            Tensor::from_data(TensorData::new((1..=15).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 5]), &dev));
        let bias: NamedTensor<B, dims![N], 1> = NamedTensor::new(
            Tensor::from_data([0.1f32, 0.2, 0.3, 0.4, 0.5], &dev));
        let out: NamedTensor<B, dims![M, N], 2> = add(mat, bias);
        println!("2. add (M,N)+(N) → (M,N)\n   {}\n", out);
    }

    // 3. add: disjoint dims (outer broadcast)
    {
        let row: NamedTensor<B, dims![M], 1> = NamedTensor::new(
            Tensor::from_data([1.0f32, 2.0, 3.0], &dev));
        let col: NamedTensor<B, dims![N], 1> = NamedTensor::new(
            Tensor::from_data([10.0f32, 20.0, 30.0, 40.0, 50.0], &dev));
        let out: NamedTensor<B, dims![M, N], 2> = add(row, col);
        println!("3. add (M)+(N) → (M,N)\n   {}\n", out);
    }

    // 4. add: commuted order
    {
        let bias: NamedTensor<B, dims![N], 1> = NamedTensor::new(
            Tensor::from_data([1.0f32, 1.0, 1.0, 1.0, 1.0], &dev));
        let mat: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(
            Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);
        let out: NamedTensor<B, dims![N, M], 2> = add(bias, mat);
        println!("4. add (N)+(M,N) → (N,M) shape={:?} mean={:.1}\n",
            out.shape().dims, Into::<f32>::into(out.inner.mean().into_scalar()));
    }

    // 5. matmul 2-D standard
    {
        let lhs_data: Vec<f32> = (0..3).flat_map(|r| vec![(r+1) as f32; 4]).collect();
        let lhs: NamedTensor<B, dims![M, K], 2> = NamedTensor::new(
            Tensor::from_data(TensorData::new(lhs_data, [3usize, 4]), &dev));
        let rhs_data: Vec<f32> = (0..4).flat_map(|_| (1..=5).map(|c| c as f32 * 0.1).collect::<Vec<_>>()).collect();
        let rhs: NamedTensor<B, dims![K, N], 2> = NamedTensor::new(
            Tensor::from_data(TensorData::new(rhs_data, [4usize, 5]), &dev));
        let c: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs, K);
        println!("5. matmul (M,K)×(K,N) → (M,N)\n   {}\n", c);
    }

    // 6. matmul 2-D with K at non-standard positions
    {
        let lhs: NamedTensor<B, dims![K, M], 2> = NamedTensor::new(
            Tensor::from_data(TensorData::new((1..=12).map(|x| x as f32).collect::<Vec<_>>(), [4usize, 3]), &dev));
        let rhs: NamedTensor<B, dims![N, K], 2> = NamedTensor::new(
            Tensor::from_data(TensorData::new((1..=20).map(|x| x as f32 * 0.1).collect::<Vec<_>>(), [5usize, 4]), &dev));
        let c: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs, K);
        println!("6. matmul (K,M)×(N,K) → (M,N)\n   {}\n", c);
    }

    // 7. matmul 3-D batched
    {
        let lhs: NamedTensor<B, dims![Batch, M, K], 3> = NamedTensor::new(
            Tensor::from_data(TensorData::new((1..=24).map(|x| x as f32).collect::<Vec<_>>(), [2usize, 3, 4]), &dev));
        let rhs: NamedTensor<B, dims![Batch, K, N], 3> = NamedTensor::new(
            Tensor::from_data(TensorData::new((1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(), [2usize, 4, 5]), &dev));
        let out: NamedTensor<B, dims![Batch, M, N], 3> = matmul(lhs, rhs, K);
        println!("7. matmul batched → shape={:?}\n", out.shape().dims);
    }

    // 8. matmul 3-D with K in middle
    {
        let lhs: NamedTensor<B, dims![M, K, Batch], 3> = NamedTensor::new(
            Tensor::from_data(TensorData::new((1..=24).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 4, 2]), &dev));
        let rhs: NamedTensor<B, dims![Batch, K, N], 3> = NamedTensor::new(
            Tensor::from_data(TensorData::new((1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(), [2usize, 4, 5]), &dev));
        let out: NamedTensor<B, dims![M, Batch, N], 3> = matmul(lhs, rhs, K);
        println!("8. matmul K-middle → shape={:?}\n", out.shape().dims);
    }

    // 9. dot product
    {
        let u: NamedTensor<B, dims![Features], 1> = NamedTensor::new(
            Tensor::from_data([1.0f32, 2.0, 3.0, 4.0], &dev));
        let v: NamedTensor<B, dims![Features], 1> = NamedTensor::new(
            Tensor::from_data([0.25f32, 0.5, 0.75, 1.0], &dev));
        println!("9. dot = {:.4}\n", dot(u, v));
    }

    // 10. sum_dim + rename
    {
        let t: NamedTensor<B, dims![SeqLen, Features], 2> = NamedTensor::new(
            Tensor::ones(Shape::new([4usize, 8]), &dev));
        let s: NamedTensor<B, dims![Features], 1> = sum_dim::<B, SeqLen, _, _, _, 2, 1>(t, 0);
        let h: NamedTensor<B, dims![Hidden], 1> = rename::<B, Features, Hidden, _, _, _, 1>(s);
        println!("10. sum_dim + rename → (Hidden=8)\n    {}\n", h);
    }
}
