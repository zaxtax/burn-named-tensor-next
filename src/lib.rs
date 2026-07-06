pub mod slice;
pub mod typed;
pub mod untyped;

pub use untyped::*;

// Re-export types needed by the `dim!` and `dims!` macros.
pub use typed::{DCons, DNil, DimName};

// Re-export types needed by the `s!` macro.
pub use slice::{DimSlice, Slice, StrSlice};
