use burn::backend::Flex;
use burn::tensor::{Shape, Tensor};
use named_tensor::{NamedTensor, matmul, s};

type B = Flex<f32>;

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
    let bias = NamedTensor::<B, 1>::new(["Vocab"], Tensor::zeros(Shape::new([1000]), &dev));
    let out: NamedTensor<B, 3> = logits + bias;

    // Index by name at runtime: slice keeps the dims, isel_by drops one
    let recent = out.clone().slice(s!["SeqLen" => 5..10]);
    let first: NamedTensor<B, 2> = out.isel_by("Batch", 0);

    println!("dims: {:?}, shape: {:?}", recent.names(), recent.shape());
    println!("dims: {:?}, shape: {:?}", first.names(), first.shape());
}
