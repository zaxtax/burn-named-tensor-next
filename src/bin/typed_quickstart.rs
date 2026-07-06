use burn::backend::Flex;
use burn::tensor::{Shape, Tensor};
use named_tensor::typed::{NamedTensor, matmul};
use named_tensor::{dim, dims, s};

type B = Flex<f32>;

dim!(Batch, SeqLen, Hidden, Vocab);

fn main() {
    let dev = Default::default();

    // Create named tensors — the type *is* the documentation
    let x: NamedTensor<B, dims![Batch, SeqLen, Hidden], 3> =
        NamedTensor::new(Tensor::ones(Shape::new([2, 10, 64]), &dev));
    let w: NamedTensor<B, dims![Hidden, Vocab], 2> =
        NamedTensor::new(Tensor::ones(Shape::new([64, 1000]), &dev));

    // matmul contracts over `Hidden` (shared, not in output) — result is dims![Batch, SeqLen, Vocab]
    let logits: NamedTensor<B, dims![Batch, SeqLen, Vocab], 3> = matmul(x, w);

    // Element-wise ops check that dims match at compile time
    let bias: NamedTensor<B, dims![Vocab], 1> =
        NamedTensor::new(Tensor::zeros(Shape::new([1000]), &dev));
    let out: NamedTensor<B, dims![Batch, SeqLen, Vocab], 3> = logits + bias; // broadcasts Vocab into the output

    // Index by name: slice keeps the dims, isel_by drops one
    let recent = out.clone().slice(s![SeqLen => 5..10]);
    let first: NamedTensor<B, dims![SeqLen, Vocab], 2> = out.isel_by(Batch, 0);

    println!(
        "dims: {:?}, shape: {:?}",
        recent.dim_names(),
        recent.shape()
    );
    println!("dims: {:?}, shape: {:?}", first.dim_names(), first.shape());
}
