#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum IMBEError {
    Derp
}

pub type IMBEResult<T> = Result<T, IMBEError>;
