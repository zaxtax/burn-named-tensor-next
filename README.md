# named-tensor-typed

Type-level [named tensors](https://nlp.seas.harvard.edu/NamedTensor) for Rust, built on [burn](https://github.com/tracel-ai/burn). Dimension names are zero-sized marker types, and all constraints — which dims a tensor carries, how operations combine them, which dim to contract over — are enforced **at compile time** through trait bounds on type-level lists.

The idea of named tensor axes was popularized by [Sasha Rush's "Tensor Considered Harmful"](https://nlp.seas.harvard.edu/NamedTensor) (2019) and has appeared in [PyTorch named tensors](https://pytorch.org/docs/stable/named_tensor.html), [tsensor](https://github.com/parrt/tensor-sensor), and Haskell's [Naperian functors](https://www.cs.ox.ac.uk/people/jeremy.gibbons/publications/aplicative.pdf). This crate takes the approach furthest: dimension constraints are checked entirely by the Rust type system, with zero runtime overhead for the naming layer itself.

## Core idea

Instead of tracking tensor axes by position (`Tensor<B, 3>` with axis 0 = batch, axis 1 = sequence, etc.), each axis gets a named marker type. The compiler rejects programs that misuse dimensions — no runtime errors, no transposition bugs.

```rust
use named_tensor::{dim, dims, NamedTensor};

dim!(Batch, SeqLen, Hidden);

// The type *is* the documentation:
let x: NamedTensor<B, dims![Batch, SeqLen, Hidden], 3> = NamedTensor::new(raw_tensor);
```

## How the type-level machinery works

### 1. Dimension markers via `DimName`

The `dim!` macro creates zero-sized types implementing `DimName`:

```rust
pub trait DimName {
    const NAME: &'static str;
}

dim!(Batch, SeqLen, Hidden);
// expands to:
// pub struct Batch;
// impl DimName for Batch { const NAME: &'static str = "Batch"; }
// ... etc.
```

These are [zero-sized types](https://doc.rust-lang.org/nomicon/exotic-sizes.html#zero-sized-types-zsts) — they carry no runtime data and are erased during compilation.

### 2. Type-level dimension lists (`DCons` / `DNil`)

Dimension names are collected into a heterogeneous type-level list, analogous to [`frunk::HList`](https://docs.rs/frunk/latest/frunk/hlist/index.html) — a Rust port of Haskell's [HList](https://hackage.haskell.org/package/HList) idiom for [heterogeneous collections at the type level](https://okmij.org/ftp/Haskell/HList-ext.pdf):

```rust
pub struct DNil;                          // empty list
pub struct DCons<H, T>(PhantomData<...>); // head + tail

// dims![Batch, SeqLen, Hidden] expands to:
// DCons<Batch, DCons<SeqLen, DCons<Hidden, DNil>>>
```

`Rank` is computed structurally — `DNil` has rank 0, each `DCons` adds 1. This connects the type-level list to burn's `const D: usize` generic.

### 3. The "frunk index trick" for overlapping impls

The key challenge: given `DCons<Batch, DCons<SeqLen, DNil>>`, how does the compiler know that `SeqLen` is contained in this list? A naive `Contains` trait would have overlapping impls (is `SeqLen` the head? or in the tail?).

The solution uses **index witness types** `Here` and `There<I>` — type-level [Peano numerals](https://en.wikipedia.org/wiki/Peano_axioms) (`Here` = zero, `There<I>` = successor). This is the same technique used by [frunk's index resolution](https://docs.rs/frunk/latest/frunk/indices/index.html):

```rust
pub struct Here;
pub struct There<I>(PhantomData<I>);

pub trait Contains<D, Idx> {}

// D is the head → index is Here
impl<D, T> Contains<D, Here> for DCons<D, T> {}

// D is in the tail → index is There<I>
impl<H, D, T, I> Contains<D, There<I>> for DCons<H, T>
where T: Contains<D, I> {}
```

The `Idx` parameter disambiguates the two impls. The compiler infers the index automatically — users never write `Here` or `There` directly. For `DCons<Batch, DCons<SeqLen, DNil>>`:
- `Contains<Batch, Here>` — Batch is at position 0
- `Contains<SeqLen, There<Here>>` — SeqLen is at position 1

If you try to use a dim that isn't in the list, **there is no valid `Idx` and compilation fails**.

### 4. `Remove` — type-level list deletion

```rust
pub trait Remove<D, Idx> { type Output; }
```

Removes a named dim from the list, producing a new list type. Used by `matmul` to express "the output has all dims from both inputs *except* the contracted dim":

```rust
// If SL = dims![M, K] and we Remove<K, _>, Output = dims![M]
```

### 5. `Subset` and `IsUnionOf` — broadcasting constraints

`add` needs to express: "the output dims are the union of the two inputs' dims." This is encoded as:

```rust
pub trait Subset<Out, Idx> {}   // every dim in Self is also in Out
pub trait IsUnionOf<SL, SR, Idx> {}  // Out ⊇ SL and Out ⊇ SR
```

The compiler checks that both inputs' dimension lists are subsets of the output's list. If you annotate the wrong output type, the subset check fails at compile time.

### 6. `ReplaceFirst` — zero-cost renaming

```rust
pub trait ReplaceFirst<Old, New, Idx> { type Output; }
```

Swaps one dim marker for another in the type-level list. The `rename` function uses this — it changes the type but emits zero machine code:

```rust
let s: NamedTensor<B, dims![Features], 1> = sum_dim(...);
let h: NamedTensor<B, dims![Hidden], 1> = rename::<B, Features, Hidden, _, _, _, 1>(s);
// No runtime work — just a type change.
```

## What the compiler catches

### Mismatched dimensions in element-wise ops

```rust
dim!(M, N, P);

let a: NamedTensor<B, dims![M, N], 2> = ...;
let b: NamedTensor<B, dims![M, P], 2> = ...;

// ERROR: P is not in dims![M, N], so IsUnionOf cannot be satisfied
let c: NamedTensor<B, dims![M, N], 2> = add(a, b);
```

### Wrong contraction dim in matmul

```rust
let lhs: NamedTensor<B, dims![M, K], 2> = ...;
let rhs: NamedTensor<B, dims![K, N], 2> = ...;

// ERROR: Contains<N, _> is not satisfied for dims![M, K]
let c: NamedTensor<B, dims![M, K], 2> = matmul(lhs, rhs, N);
```

### Dot product on different dims

```rust
dim!(Features, Time);

let u: NamedTensor<B, dims![Features], 1> = ...;
let v: NamedTensor<B, dims![Time], 1> = ...;

// ERROR: dims![Features] ≠ dims![Time], so the S parameter can't unify
dot(u, v);
```

### Removing a dim that doesn't exist

```rust
let t: NamedTensor<B, dims![M, N], 2> = ...;

// ERROR: Contains<K, _> not satisfied for dims![M, N]
let s = sum_dim::<B, K, _, _, _, 2, 1>(t, 0);
```

## Operations and their type-level contracts

| Operation | Constraint | What it means |
|-----------|-----------|---------------|
| `add(lhs, rhs)` | `Out: IsUnionOf<SL, SR>` | Output dims must be the union of both inputs |
| `matmul(lhs, rhs, K)` | `SL: Contains<K>`, `SR: Contains<K>`, `Out: IsUnionOf<SL\K, SR\K>` | Both inputs must have K; output is the union of the remaining dims |
| `dot(a, b)` | Same type `S` for both | Both vectors must name the same dimension |
| `sum_dim(t)` | `S: Contains<C>`, `S: Remove<C, Output=Out>` | The summed dim must exist; output type has it removed |
| `rename(t)` | `S: Contains<Old>`, `S: ReplaceFirst<Old, New, Output=Out>` | Old dim must exist; output type has it swapped |

## Running the demos

Typed API — dim names are marker types, all checks at compile time:

```sh
cargo run --bin typed_demo
```

Untyped API — the same operations with `&str` dim names, checked at run time. Exposed under `named_tensor::untyped`:

```sh
cargo run --bin untyped_demo
```

The untyped module mirrors the typed surface (`NamedTensor`, `add`, `matmul`, `dot`, `sum`, `permute`, `rename`) but swaps type-level dim markers for string literals — useful when dim names are only known at run time, or as a gentler on-ramp before reaching for the type-level version.

## How this differs from prior work

Most existing approaches to compile-time tensor safety rely on **dependent type systems** (Haskell's `DataKinds`/type families, Idris, Coq). This crate works on **stable Rust** with no language extensions, no procedural macros, and no dependent types — it encodes dimension names as zero-sized marker types and resolves overlapping trait impls via the frunk index trick.

- **[TensorSafe](https://dl.acm.org/doi/10.1145/3355378.3355379)** (Cedroni & Uchoa, SBLP 2019; [Hackage](https://hackage.haskell.org/package/tensor-safe)) — validates DNN layer shapes at compile time in Haskell using `DataKinds` and type families. Operates on *positional* shapes (`'[28, 28, 1]`), not named dimensions, and targets network architecture validation rather than individual tensor operations.
- **[Hasktorch typed tensors](https://hasktorch.github.io/tutorial/07-typed-tensors.html)** ([repo](https://github.com/hasktorch/hasktorch)) — lifts shape, dtype, and device into Haskell's type system via `DataKinds`. Shapes are positional `Nat` lists, not named axes. Requires GHC extensions (`DataKinds`, `TypeFamilies`, `GADTs`) that have no Rust equivalent.
- **[Gradual Tensor Shape Checking](https://arxiv.org/abs/2203.08402)** (Castagna et al., ESOP 2023) — proves that general tensor shape inference is *undecidable* and proposes a gradual typing approach: infer what you can statically, insert dynamic checks for the rest. This crate sidesteps undecidability by requiring the user to annotate output dimension lists explicitly — the compiler then *verifies* the annotation rather than inferring it.
- **[torchtyping](https://github.com/patrick-kidger/torchtyping)** (Kidger) — Python runtime annotations for tensor shape, dtype, and axis names. Checks happen at runtime via `typeguard`; errors are caught during execution, not compilation.

## See also

- [Tensor Considered Harmful](https://nlp.seas.harvard.edu/NamedTensor) — Sasha Rush's original named tensor manifesto (2019)
- [PyTorch Named Tensors](https://pytorch.org/docs/stable/named_tensor.html) — runtime named dimensions in PyTorch
- [Naperian Functors](https://www.cs.ox.ac.uk/people/jeremy.gibbons/publications/aplicative.pdf) — Gibbons (2016), the Haskell-side theory of shape-indexed containers
- [frunk](https://github.com/lloydmeta/frunk) — the Rust HList library whose index trick this crate adapts
- [typenum](https://docs.rs/typenum) — type-level arithmetic in Rust, a complementary approach to compile-time dimension checking
- [Strongly Typed Heterogeneous Collections](https://okmij.org/ftp/Haskell/HList-ext.pdf) — Kiselyov, Lammel, Schupke (2004), the foundational HList paper
