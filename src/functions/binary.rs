
pub trait BinaryOp {
    type Other;
    type Output;
    fn add(&self, other: Self::Other) -> Self::Output;
    fn sub(&self, other: Self::Other) -> Self::Output;
}

