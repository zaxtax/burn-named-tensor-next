use burn::tensor::{Shape, Tensor};
use named_tensor::{NamedTensor, matmul, s};


fn main() {
    let dev = Default::default();

    let x = NamedTensor::<3>::new(
        ["Batch", "SeqLen", "Hidden"],
        Tensor::ones(Shape::new([2, 10, 64]), &dev),
    );
    let w = NamedTensor::<2>::new(
        ["Hidden", "Vocab"],
        Tensor::ones(Shape::new([64, 1000]), &dev),
    );

    let logits: NamedTensor<3> = matmul(x, w, "Hidden");
    let bias = NamedTensor::<1>::new(["Vocab"], Tensor::zeros(Shape::new([1000]), &dev));
    let out: NamedTensor<3> = logits + bias;

    // Index by name at runtime: slice keeps the dims, isel_by drops one
    let recent = out.clone().slice(s!["SeqLen" => 5..10]);
    let head = out.clone().slice_by("SeqLen", 0..4);
    let first: NamedTensor<2> = out.clone().isel_by("Batch", 0);

    // Write to named regions; slice_assign aligns `values` axes by dim name
    let masked = out.slice_fill(s!["SeqLen" => 5..10], 0.0);
    let patched = masked.slice_assign(s!["SeqLen" => 5..10], recent.clone());

    println!("dims: {:?}, shape: {:?}", recent.names(), recent.shape());
    println!("dims: {:?}, shape: {:?}", head.names(), head.shape());
    println!("dims: {:?}, shape: {:?}", first.names(), first.shape());
    println!("dims: {:?}, shape: {:?}", patched.names(), patched.shape());
}
