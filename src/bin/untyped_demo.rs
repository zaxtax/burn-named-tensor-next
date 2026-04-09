use burn::backend::NdArray;
use burn::tensor::{Shape, Tensor, TensorData};
use named_tensor::untyped::{NamedTensor, add, dot, matmul, permute, rename, sum};

type B = NdArray<f32>;

fn main() {
    let dev: <B as burn::prelude::Backend>::Device = Default::default();

    println!("=== named-tensor untyped demo ===\n");

    // 1. add same shape
    {
        let a: NamedTensor<B, 2> =
            NamedTensor::new(["M", "N"], Tensor::ones(Shape::new([3usize, 5]), &dev));
        let b: NamedTensor<B, 2> = NamedTensor::new(
            ["M", "N"],
            Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
        );
        let c: NamedTensor<B, 2> = add(a, b);
        println!(
            "1. add (M,N)+(M,N) → mean={:.1}\n",
            Into::<f32>::into(c.inner.mean().into_scalar())
        );
    }

    // 2. add rank-2 + rank-1 broadcast
    {
        let mat: NamedTensor<B, 2> = NamedTensor::new(
            ["M", "N"],
            Tensor::from_data(
                TensorData::new((1..=15).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 5]),
                &dev,
            ),
        );
        let bias: NamedTensor<B, 1> =
            NamedTensor::new(["N"], Tensor::from_data([0.1f32, 0.2, 0.3, 0.4, 0.5], &dev));
        let out: NamedTensor<B, 2> = add(mat, bias);
        println!("2. add (M,N)+(N) → (M,N)\n   {}\n", out);
    }

    // 3. add disjoint dims (outer broadcast)
    {
        let row: NamedTensor<B, 1> =
            NamedTensor::new(["M"], Tensor::from_data([1.0f32, 2.0, 3.0], &dev));
        let col: NamedTensor<B, 1> = NamedTensor::new(
            ["N"],
            Tensor::from_data([10.0f32, 20.0, 30.0, 40.0, 50.0], &dev),
        );
        let out: NamedTensor<B, 2> = add(row, col);
        println!("3. add (M)+(N) → (M,N)\n   {}\n", out);
    }

    // 4. add commuted order
    {
        let bias: NamedTensor<B, 1> =
            NamedTensor::new(["N"], Tensor::from_data([1.0f32, 1.0, 1.0, 1.0, 1.0], &dev));
        let mat: NamedTensor<B, 2> = NamedTensor::new(
            ["M", "N"],
            Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0,
        );
        let out: NamedTensor<B, 2> = add(bias, mat);
        println!(
            "4. add (N)+(M,N) → (N,M) shape={:?} mean={:.1}\n",
            out.shape().dims,
            Into::<f32>::into(out.inner.mean().into_scalar())
        );
    }

    // 5. matmul 2-D standard
    {
        let lhs_data: Vec<f32> = (0..3).flat_map(|r| vec![(r + 1) as f32; 4]).collect();
        let lhs: NamedTensor<B, 2> = NamedTensor::new(
            ["M", "K"],
            Tensor::from_data(TensorData::new(lhs_data, [3usize, 4]), &dev),
        );
        let rhs_data: Vec<f32> = (0..4)
            .flat_map(|_| (1..=5).map(|c| c as f32 * 0.1).collect::<Vec<_>>())
            .collect();
        let rhs: NamedTensor<B, 2> = NamedTensor::new(
            ["K", "N"],
            Tensor::from_data(TensorData::new(rhs_data, [4usize, 5]), &dev),
        );
        let c: NamedTensor<B, 2> = matmul(lhs, rhs, "K");
        println!("5. matmul (M,K)×(K,N) → (M,N)\n   {}\n", c);
    }

    // 6. matmul 2-D with K at non-standard positions
    {
        let lhs: NamedTensor<B, 2> = NamedTensor::new(
            ["K", "M"],
            Tensor::from_data(
                TensorData::new((1..=12).map(|x| x as f32).collect::<Vec<_>>(), [4usize, 3]),
                &dev,
            ),
        );
        let rhs: NamedTensor<B, 2> = NamedTensor::new(
            ["N", "K"],
            Tensor::from_data(
                TensorData::new(
                    (1..=20).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                    [5usize, 4],
                ),
                &dev,
            ),
        );
        let c: NamedTensor<B, 2> = matmul(lhs, rhs, "K");
        println!("6. matmul (K,M)×(N,K) → (M,N)\n   {}\n", c);
    }

    // 7. matmul 3-D batched
    {
        let lhs: NamedTensor<B, 3> = NamedTensor::new(
            ["Batch", "M", "K"],
            Tensor::from_data(
                TensorData::new(
                    (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                    [2usize, 3, 4],
                ),
                &dev,
            ),
        );
        let rhs: NamedTensor<B, 3> = NamedTensor::new(
            ["Batch", "K", "N"],
            Tensor::from_data(
                TensorData::new(
                    (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                    [2usize, 4, 5],
                ),
                &dev,
            ),
        );
        let out: NamedTensor<B, 3> = matmul(lhs, rhs, "K");
        println!(
            "7. matmul batched → names={:?} shape={:?}\n",
            out.names(),
            out.shape().dims
        );
    }

    // 8. matmul 3-D with K in middle
    {
        let lhs: NamedTensor<B, 3> = NamedTensor::new(
            ["M", "K", "Batch"],
            Tensor::from_data(
                TensorData::new(
                    (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                    [3usize, 4, 2],
                ),
                &dev,
            ),
        );
        let rhs: NamedTensor<B, 3> = NamedTensor::new(
            ["Batch", "K", "N"],
            Tensor::from_data(
                TensorData::new(
                    (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                    [2usize, 4, 5],
                ),
                &dev,
            ),
        );
        let out: NamedTensor<B, 3> = matmul(lhs, rhs, "K");
        println!(
            "8. matmul K-middle → names={:?} shape={:?}\n",
            out.names(),
            out.shape().dims
        );
    }

    // 8b. matmul mixed-rank: (M,K) × (K,N,Batch) → (M,N,Batch)
    {
        let lhs: NamedTensor<B, 2> = NamedTensor::new(
            ["M", "K"],
            Tensor::from_data(
                TensorData::new((1..=6).map(|x| x as f32).collect::<Vec<_>>(), [3usize, 2]),
                &dev,
            ),
        );
        let rhs: NamedTensor<B, 3> = NamedTensor::new(
            ["K", "N", "Batch"],
            Tensor::from_data(
                TensorData::new(
                    (1..=40).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                    [2usize, 5, 4],
                ),
                &dev,
            ),
        );
        let out: NamedTensor<B, 3> = matmul(lhs, rhs, "K");
        println!(
            "8b. matmul mixed-rank (M,K)×(K,N,Batch) → names={:?} shape={:?}\n",
            out.names(),
            out.shape().dims
        );
    }

    // 8c. matmul double-contract (A,K1,K2)×(K2,K1,B) → (A,B)
    {
        let lhs: NamedTensor<B, 3> = NamedTensor::new(
            ["A", "K1", "K2"],
            Tensor::from_data(
                TensorData::new(
                    (1..=24).map(|x| x as f32).collect::<Vec<_>>(),
                    [4usize, 2, 3],
                ),
                &dev,
            ),
        );
        let rhs: NamedTensor<B, 3> = NamedTensor::new(
            ["K2", "K1", "B"],
            Tensor::from_data(
                TensorData::new(
                    (1..=30).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
                    [3usize, 2, 5],
                ),
                &dev,
            ),
        );
        let out: NamedTensor<B, 2> = matmul(lhs, rhs, ["K1", "K2"]);
        println!(
            "8c. matmul double-contract (A,K1,K2)×(K2,K1,B) → names={:?} shape={:?}\n    {}\n",
            out.names(),
            out.shape().dims,
            out
        );
    }

    // 9. dot product
    {
        let u: NamedTensor<B, 1> = NamedTensor::new(
            ["Features"],
            Tensor::from_data([1.0f32, 2.0, 3.0, 4.0], &dev),
        );
        let v: NamedTensor<B, 1> = NamedTensor::new(
            ["Features"],
            Tensor::from_data([0.25f32, 0.5, 0.75, 1.0], &dev),
        );
        println!("9. dot = {:.4}\n", dot(u, v));
    }

    // 10. permute (transpose)
    {
        let t: NamedTensor<B, 3> = NamedTensor::new(
            ["Batch", "M", "N"],
            Tensor::from_data(
                TensorData::new(
                    (1..=30).map(|x| x as f32).collect::<Vec<_>>(),
                    [2usize, 3, 5],
                ),
                &dev,
            ),
        );
        println!("10. permute (Batch,M,N) shape={:?}", t.shape().dims);
        let t2: NamedTensor<B, 3> = permute(t, ["N", "Batch", "M"]);
        println!("    → (N,Batch,M) shape={:?}\n", t2.shape().dims);
    }

    // 11. sum + rename
    {
        let t: NamedTensor<B, 2> = NamedTensor::new(
            ["SeqLen", "Features"],
            Tensor::ones(Shape::new([4usize, 8]), &dev),
        );
        let s: NamedTensor<B, 1> = sum(t, "SeqLen");
        let h: NamedTensor<B, 1> = rename(s, "Features", "Hidden");
        println!("11. sum + rename → (Hidden=8)\n    {}\n", h);
    }

    println!("=== done ===");
}
