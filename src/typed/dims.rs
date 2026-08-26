use burn::prelude::*;
use std::marker::PhantomData;

pub trait DimName {
    const NAME: &'static str;
}

#[macro_export]
macro_rules! dim {
    ($($name:ident),+ $(,)?) => {
        $(
            #[derive(Clone, Copy, Debug)]
            pub struct $name;
            impl $crate::DimName for $name {
                const NAME: &'static str = stringify!($name);
            }
        )+
    };
}

pub struct Here;
pub struct There<I>(PhantomData<I>);

/// Marker: this dim is shared with the other tensor (will be contracted).
pub struct IsShared;
/// Marker: this dim is exclusive to this tensor (will be kept).
pub struct IsExclusive;

pub struct DNil;
pub struct DCons<H, T>(PhantomData<(H, T)>);

#[macro_export]
macro_rules! dims {
    ()                  => { $crate::DNil };
    ($h:ty)             => { $crate::DCons<$h, $crate::DNil> };
    ($h:ty, $($t:ty),+) => { $crate::DCons<$h, $crate::dims![$($t),+]> };
}

pub trait Rank {
    const RANK: usize;
}
impl Rank for DNil {
    const RANK: usize = 0;
}
impl<H, T: Rank> Rank for DCons<H, T> {
    const RANK: usize = 1 + T::RANK;
}

pub trait NameList {
    fn names() -> Vec<&'static str>;
}
impl NameList for DNil {
    fn names() -> Vec<&'static str> {
        vec![]
    }
}
impl<H: DimName, T: NameList> NameList for DCons<H, T> {
    fn names() -> Vec<&'static str> {
        let mut v = vec![H::NAME];
        v.extend(T::names());
        v
    }
}

#[diagnostic::on_unimplemented(
    message = "dim `{D}` is not present in this tensor's dimension list",
    label = "dim `{D}` missing here",
    note = "double-check the dim markers on this `NamedTensor` — `{D}` must appear in its `dims![…]` list"
)]
pub trait Contains<D, Idx> {}
impl<D, T> Contains<D, Here> for DCons<D, T> {}
impl<H, D, T, I> Contains<D, There<I>> for DCons<H, T> where T: Contains<D, I> {}

#[diagnostic::on_unimplemented(
    message = "cannot remove dim `{D}` — it is not present in the dimension list",
    note = "this usually appears alongside a `Contains<{D}, _>` error on the same line; fixing the missing dim resolves both"
)]
pub trait Remove<D, Idx> {
    type Output;
}
impl<D, T> Remove<D, Here> for DCons<D, T> {
    type Output = T;
}
impl<H, D, T, I> Remove<D, There<I>> for DCons<H, T>
where
    T: Remove<D, I>,
{
    type Output = DCons<H, <T as Remove<D, I>>::Output>;
}

#[diagnostic::on_unimplemented(
    message = "dim list `{Ks}` is not fully contained in `{Self}`",
    note = "every dim in `{Ks}` must appear in `{Self}` so it can be removed"
)]
pub trait RemoveAll<Ks, Idx> {
    type Output;
}
impl<S> RemoveAll<DNil, ()> for S {
    type Output = S;
}
impl<S, KH, KT, IH, IT> RemoveAll<DCons<KH, KT>, (IH, IT)> for S
where
    S: Remove<KH, IH>,
    <S as Remove<KH, IH>>::Output: RemoveAll<KT, IT>,
{
    type Output = <<S as Remove<KH, IH>>::Output as RemoveAll<KT, IT>>::Output;
}

/// Partitions `Self` relative to `Other`, using `Out` (the expected output dim
/// list) to disambiguate: a **shared** dim appears in `Other` (and is
/// contracted away), while an **exclusive** dim appears in `Out` (and is kept).
///
/// The compiler can always pick exactly one branch because:
/// - A shared dim is in `Other` but *not* in `Out` → only `IsShared` applies.
/// - An exclusive dim is in `Out` but *not* in `Other` → only `IsExclusive` applies.
/// - If a dim is in *both* `Other` and `Out`, the user's output annotation is
///   wrong (a contracted dim shouldn't survive); the compiler reports ambiguity.
/// - If a dim is in *neither*, the output is missing a required dim; the
///   compiler reports an unsatisfied bound.
#[diagnostic::on_unimplemented(
    message = "cannot partition `{Self}` into shared/exclusive dims given `{Other}` and output `{Out}`",
    note = "each dim in `{Self}` must be shared (in `{Other}`, contracted) or exclusive (in `{Out}`, kept) — but not both and not neither"
)]
pub trait Exclusive<Other, Out, Idx> {
    type Output;
}
impl<Other, Out> Exclusive<Other, Out, ()> for DNil {
    type Output = DNil;
}
/// Dim `H` is shared with `Other` — contracted away, not in output.
impl<H, T, Other, Out, IH, IT> Exclusive<Other, Out, (IsShared, IH, IT)> for DCons<H, T>
where
    Other: Contains<H, IH>,
    T: Exclusive<Other, Out, IT>,
{
    type Output = <T as Exclusive<Other, Out, IT>>::Output;
}
/// Dim `H` is exclusive to `Self` — kept in output.
impl<H, T, Other, Out, IO, IT> Exclusive<Other, Out, (IsExclusive, IO, IT)> for DCons<H, T>
where
    Out: Contains<H, IO>,
    T: Exclusive<Other, Out, IT>,
{
    type Output = DCons<H, <T as Exclusive<Other, Out, IT>>::Output>;
}

#[diagnostic::on_unimplemented(
    message = "dimension list `{Self}` is not a subset of `{Out}`",
    label = "some dim in `{Self}` is missing from `{Out}`",
    note = "every dim in `{Self}` must also appear in `{Out}` (order doesn't matter, but the set must match or be contained)"
)]
pub trait Subset<Out, Idx> {}
impl<Out> Subset<Out, ()> for DNil {}
impl<H, T, Out, IC, IT> Subset<Out, (IC, IT)> for DCons<H, T>
where
    Out: Contains<H, IC>,
    T: Subset<Out, IT>,
{
}

#[diagnostic::on_unimplemented(
    message = "output dims `{Self}` are not a valid union of `{SL}` and `{SR}`",
    label = "output `{Self}` must contain every dim from both inputs",
    note = "the annotated output dimension list must be a superset of both `{SL}` and `{SR}` — add any missing dim from either side to your output annotation"
)]
pub trait IsUnionOf<SL, SR, Idx> {}
impl<Out, SL, SR, LIdx, RIdx> IsUnionOf<SL, SR, (LIdx, RIdx)> for Out
where
    SL: Subset<Out, LIdx>,
    SR: Subset<Out, RIdx>,
{
}

#[diagnostic::on_unimplemented(
    message = "cannot rename dim `{Old}` to `{New}` — `{Old}` is not present in the dimension list",
    note = "`rename` requires the old dim to exist. If you're trying to introduce a brand-new dim, `rename` is not the right tool — construct a fresh `NamedTensor` with the desired marker instead"
)]
pub trait ReplaceFirst<Old, New, Idx> {
    type Output;
}
impl<Old, New, T> ReplaceFirst<Old, New, Here> for DCons<Old, T> {
    type Output = DCons<New, T>;
}
impl<H, Old, New, T, I> ReplaceFirst<Old, New, There<I>> for DCons<H, T>
where
    T: ReplaceFirst<Old, New, I>,
{
    type Output = DCons<H, <T as ReplaceFirst<Old, New, I>>::Output>;
}

// ── Helpers ──

pub(crate) fn find_axis(list: &[&'static str], name: &'static str) -> usize {
    list.iter()
        .position(|&n| n == name)
        .unwrap_or_else(|| panic!("named-tensor: dim '{name}' not found in {list:?}"))
}

pub(crate) fn build_perm(from: &[&'static str], to: &[&'static str]) -> Vec<usize> {
    to.iter().map(|name| find_axis(from, name)).collect()
}

pub(crate) fn is_identity(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

pub(crate) fn permute_if_needed<B: Backend, const D: usize>(
    t: Tensor<B, D>,
    perm: &[usize],
) -> Tensor<B, D> {
    if is_identity(perm) {
        return t;
    }
    let arr: [isize; D] = std::array::from_fn(|i| perm[i] as isize);
    t.permute(arr)
}

pub(crate) fn align_to_impl<B: Backend, const D_IN: usize, const D_OUT: usize>(
    t: Tensor<B, D_IN>,
    operand_names: &[&'static str],
    target_names: &[&'static str],
) -> Tensor<B, D_OUT> {
    let missing: Vec<isize> = (0..D_OUT as isize)
        .filter(|&i| !operand_names.contains(&target_names[i as usize]))
        .collect();

    let expanded: Tensor<B, D_OUT> = t.unsqueeze_dims(&missing);

    let mut src = operand_names.iter().copied();
    let current: Vec<&'static str> = (0..D_OUT)
        .map(|i| {
            if missing.contains(&(i as isize)) {
                target_names[i]
            } else {
                src.next().unwrap()
            }
        })
        .collect();

    permute_if_needed(expanded, &build_perm(&current, target_names))
}
