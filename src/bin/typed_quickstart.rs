use burn::backend::NdArray;
use burn::tensor::{Shape, Tensor};
use named_tensor::typed::{matmul, NamedTensor};
use named_tensor::{dim, dims};

type B = NdArray<f32>;

dim!(Batch, SeqLen, Hidden, Vocab);

fn main() {
    let dev = Default::default();

    // Create named tensors — the type *is* the documentation
    let x: NamedTensor<B, dims![Batch, SeqLen, Hidden], 3> =
        NamedTensor::new(Tensor::ones(Shape::new([2, 10, 64]), &dev));
    let w: NamedTensor<B, dims![Hidden, Vocab], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([64, 1000]), &dev));

    // matmul contracts over `Hidden` — result is dims![Batch, SeqLen, Vocab]
    let logits: NamedTensor<B, dims![Batch, SeqLen, Vocab], 3> =
        matmul(x, w, Hidden);

    // Element-wise ops check that dims match at compile time
    let bias: NamedTensor<B, dims![Vocab], 1> =
        NamedTensor::new(Tensor::zeros(Shape::new([1000]), &dev));
    let out: NamedTensor<B, dims![Batch, SeqLen, Vocab], 3> =
        logits + bias; // broadcasts Vocab into the output

    println!("dims: {:?}, shape: {:?}", out.dim_names(), out.shape());
}
