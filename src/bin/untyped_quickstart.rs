use burn::backend::NdArray;
use burn::tensor::{Shape, Tensor};
use named_tensor::{matmul, NamedTensor};

type B = NdArray<f32>;

fn main() {
    let dev = Default::default();

    let x = NamedTensor::<B, 3>::new(
        ["Batch", "SeqLen", "Hidden"],
        Tensor::ones(Shape::new([2, 10, 64]), &dev),
    );
    let w = NamedTensor::<B, 2>::new(
        ["Hidden", "Vocab"],
        Tensor::ones(Shape::new([64, 1000]), &dev),
    );

    let logits: NamedTensor<B, 3> = matmul(x, w, "Hidden");
    let bias = NamedTensor::<B, 1>::new(
        ["Vocab"],
        Tensor::zeros(Shape::new([1000]), &dev),
    );
    let out: NamedTensor<B, 3> = logits + bias;

    println!("dims: {:?}, shape: {:?}", out.names(), out.shape());
}
