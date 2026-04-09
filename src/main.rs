use named_tensor::{dim, dims, NamedTensor, add, matmul, dot, sum_dim, rename};
use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData, Shape};

// Declare all dim markers + their runtime name strings in one line.
dim!(Batch, M, K, N, Features, SeqLen, Hidden);

type B = NdArray<f32>;

fn main() {
    let dev: <B as burn::prelude::Backend>::Device = Default::default();

    println!("══════════════════════════════════════════════════════");
    println!("  named-tensor demo");
    println!("══════════════════════════════════════════════════════\n");

    // ── 1. add: same shape — union(S, S) = S ──────────────────────────────
    {
        let a: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(
            Tensor::ones(Shape::new([3usize, 5]), &dev));
        let b: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(
            Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);

        let c: NamedTensor<B, dims![M, N], 2> = add(a, b);
        let mean: f32 = c.inner.mean().into_scalar().into();
        println!("── 1. add same shape  (M,N)+(M,N) → (M,N)");
        println!("   mean (expect 3.0): {mean:.1}\n");
    }

    // ── 2. add: rank-2 + rank-1 — union({M,N},{N}) = {M,N} ───────────────
    // N is shared; M is only in lhs → rhs gets a size-1 M axis inserted.
    {
        let mat: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(
            Tensor::from_data(
                TensorData::new((1..=15).map(|x| x as f32).collect::<Vec<_>>(), [3usize,5]),
                &dev));
        let bias: NamedTensor<B, dims![N], 1> = NamedTensor::new(
            Tensor::from_data([0.1f32, 0.2, 0.3, 0.4, 0.5], &dev));

        // union(dims![M,N], dims![N]) = dims![M,N]
        // bias gets a size-1 axis at position 0 (M), becoming [1,5] → broadcast.
        let out: NamedTensor<B, dims![M, N], 2> = add(mat, bias);
        println!("── 2. add rank-2 + rank-1  (M=3,N=5)+(N=5) → (M=3,N=5)");
        println!("   {}\n", out);
    }

    // ── 3. add: disjoint dims — union({M},{N}) = {M,N} (outer broadcast) ──
    // Both get a size-1 axis inserted for the other's dim → outer product sum.
    {
        let row: NamedTensor<B, dims![M], 1> = NamedTensor::new(
            Tensor::from_data([1.0f32, 2.0, 3.0], &dev));
        let col: NamedTensor<B, dims![N], 1> = NamedTensor::new(
            Tensor::from_data([10.0f32, 20.0, 30.0, 40.0, 50.0], &dev));

        // union(dims![M], dims![N]) = dims![M,N]   (M kept, N appended)
        let out: NamedTensor<B, dims![M, N], 2> = add(row, col);
        println!("── 3. add disjoint dims  (M=3)+(N=5) → (M=3,N=5)  [outer broadcast sum]");
        println!("   {}\n", out);
    }

    // ── 4. add: rank-1 + rank-2 commuted — union({N},{M,N}) = {N,M} ───────
    {
        let bias: NamedTensor<B, dims![N], 1> = NamedTensor::new(
            Tensor::from_data([1.0f32, 1.0, 1.0, 1.0, 1.0], &dev));
        let mat: NamedTensor<B, dims![M, N], 2> = NamedTensor::new(
            Tensor::ones(Shape::new([3usize, 5]), &dev) * 2.0);

        // union(dims![N], dims![M,N]) = dims![N, M]  (N first, M appended)
        let out: NamedTensor<B, dims![N, M], 2> = add(bias, mat);
        println!("── 4. add commuted  (N=5)+(M=3,N=5) → (N=5,M=3)");
        println!("   shape {:?}  mean (expect 3.0): {:.1}\n",
            out.shape().dims, Into::<f32>::into(out.inner.mean().into_scalar()));
    }

    // ── 5. matmul 2-D: K at standard positions ────────────────────────────
    {
        let lhs_data: Vec<f32> = (0..3).flat_map(|r| vec![(r+1) as f32; 4]).collect();
        let lhs: NamedTensor<B, dims![M, K], 2> = NamedTensor::new(
            Tensor::from_data(TensorData::new(lhs_data, [3usize,4]), &dev));

        let rhs_data: Vec<f32> = (0..4).flat_map(|_| (1..=5).map(|c| c as f32*0.1)
            .collect::<Vec<_>>()).collect();
        let rhs: NamedTensor<B, dims![K, N], 2> = NamedTensor::new(
            Tensor::from_data(TensorData::new(rhs_data, [4usize,5]), &dev));

        // union(dims![M,K], dims![K,N]) − {K} = dims![M,N]
        let c: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs, K);
        println!("── 5. matmul 2-D standard  (M=3,K=4)×(K=4,N=5) → (M=3,N=5)");
        println!("   {}\n", c);
    }

    // ── 6. matmul 2-D: K at non-standard positions ────────────────────────
    // lhs=(K=4,M=3), rhs=(N=5,K=4) — K is first in lhs, last in rhs.
    // Runtime finds K at position 0 and 1 respectively, permutes, contracts.
    {
        let lhs: NamedTensor<B, dims![K, M], 2> = NamedTensor::new(
            Tensor::from_data(
                TensorData::new((1..=12).map(|x| x as f32).collect::<Vec<_>>(), [4usize,3]),
                &dev));
        let rhs: NamedTensor<B, dims![N, K], 2> = NamedTensor::new(
            Tensor::from_data(
                TensorData::new((1..=20).map(|x| x as f32 * 0.1).collect::<Vec<_>>(), [5usize,4]),
                &dev));

        // K found at lhs[0], rhs[1] → permuted to (...,M,K)×(...,K,N) at runtime
        // union(dims![K,M], dims![N,K]) − {K} = dims![M,N]
        let c: NamedTensor<B, dims![M, N], 2> = matmul(lhs, rhs, K);
        println!("── 6. matmul 2-D K non-standard  (K=4,M=3)×(N=5,K=4) → (M=3,N=5)");
        println!("   {}\n", c);
    }

    // ── 7. matmul 3-D batched: standard ───────────────────────────────────
    {
        let lhs: NamedTensor<B, dims![Batch, M, K], 3> = NamedTensor::new(
            Tensor::from_data(
                TensorData::new((1..=24).map(|x| x as f32).collect::<Vec<_>>(), [2usize,3,4]),
                &dev));
        let rhs: NamedTensor<B, dims![Batch, K, N], 3> = NamedTensor::new(
            Tensor::from_data(
                TensorData::new((1..=40).map(|x| x as f32*0.1).collect::<Vec<_>>(), [2usize,4,5]),
                &dev));

        let out: NamedTensor<B, dims![Batch, M, N], 3> = matmul(lhs, rhs, K);
        println!("── 7. matmul 3-D batched  (Batch=2,M=3,K=4)×(Batch=2,K=4,N=5) → (Batch=2,M=3,N=5)");
        println!("   shape: {:?}\n", out.shape().dims);
    }

    // ── 8. matmul 3-D: K in the middle of both ────────────────────────────
    // lhs=(M=3,K=4,Batch=2), rhs=(Batch=2,K=4,N=5)
    // K is at position 1 in both. Runtime permutes to (...,M,K)×(...,K,N).
    {
        let lhs: NamedTensor<B, dims![M, K, Batch], 3> = NamedTensor::new(
            Tensor::from_data(
                TensorData::new((1..=24).map(|x| x as f32).collect::<Vec<_>>(), [3usize,4,2]),
                &dev));
        let rhs: NamedTensor<B, dims![Batch, K, N], 3> = NamedTensor::new(
            Tensor::from_data(
                TensorData::new((1..=40).map(|x| x as f32*0.1).collect::<Vec<_>>(), [2usize,4,5]),
                &dev));

        // union(dims![M,K,Batch], dims![Batch,K,N]) − {K} = dims![M,Batch,N]
        let out: NamedTensor<B, dims![M, Batch, N], 3> = matmul(lhs, rhs, K);
        println!("── 8. matmul K in middle  (M,K,Batch)×(Batch,K,N) → (M,Batch,N)");
        println!("   shape: {:?}\n", out.shape().dims);
    }

    // ── 9. dot product ─────────────────────────────────────────────────────
    {
        let u: NamedTensor<B, dims![Features], 1> = NamedTensor::new(
            Tensor::from_data([1.0f32, 2.0, 3.0, 4.0], &dev));
        let v: NamedTensor<B, dims![Features], 1> = NamedTensor::new(
            Tensor::from_data([0.25f32, 0.5, 0.75, 1.0], &dev));
        let r: f32 = dot(u, v);
        let expected = 1.0*0.25 + 2.0*0.5 + 3.0*0.75 + 4.0*1.0;
        println!("── 9. dot  Features(4)·Features(4)");
        println!("   {r:.4}  (expected {expected:.4})\n");
    }

    // ── 10. sum_dim + rename ───────────────────────────────────────────────
    {
        let t: NamedTensor<B, dims![SeqLen, Features], 2> = NamedTensor::new(
            Tensor::ones(Shape::new([4usize, 8]), &dev));
        // sum over SeqLen (axis 0) → (Features=8,)
        let s: NamedTensor<B, dims![Features], 1> =
            sum_dim::<B, SeqLen, _, _, _, 2, 1>(t, 0);
        // rename Features → Hidden
        let h: NamedTensor<B, dims![Hidden], 1> =
            rename::<B, Features, Hidden, _, _, _, 1>(s);
        println!("── 10. sum_dim<SeqLen> + rename Features→Hidden  → (Hidden=8,)");
        println!("   {} (each = 4.0)\n", h);
    }

    println!("══════════════════════════════════════════════════════");
    println!("  Compile-time errors (uncomment to verify):");
    println!("  • matmul where K type differs between args → Contains fails");
    println!("  • dot with different dim markers           → S type mismatch");
    println!("  • sum_dim with dim not in shape            → Contains fails");
    println!("  • add output annotated as wrong shape      → Union::Output mismatch");
    println!("══════════════════════════════════════════════════════");
}
