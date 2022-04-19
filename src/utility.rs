use std::fmt;

use itertools::free::enumerate;
use ndarray::{array, Array, ArrayD, Axis, Dimension, Ix1, Ix2, RemoveAxis};

use std::error::Error;

pub fn argmax(a: ArrayD<f64>) -> usize {
    let mut high_index = 0usize;
    let mut high = f64::MIN;
    for (i, item) in a.iter().enumerate() {
        if *item > high {
            high = *item;
            high_index = i;
        }
    }
    high_index
}

pub fn dot(x: &ArrayD<f64>, y: &ArrayD<f64>) -> ArrayD<f64> {
    let dx = x.shape().len();
    let dy = y.shape().len();

    let mut a_grad = None;
    if dx == 1 {
        let x1 = x.clone().into_dimensionality::<Ix1>().ok().unwrap();
        if dy == 1 {
            let y1 = y.clone().into_dimensionality::<Ix1>().ok().unwrap();
            a_grad = Some(array![x1.dot(&y1)].into_dyn());
        } else if dy == 2 {
            let y2 = y.clone().into_dimensionality::<Ix2>().ok().unwrap();
            a_grad = Some(x1.dot(&y2).into_dyn());
        } else {
            panic!("Illegal sized 'a' array");
        }
    } else if dx == 2 {
        let x2 = x.clone().into_dimensionality::<Ix2>().ok().unwrap();
        if dy == 1 {
            let y1 = y.clone().into_dimensionality::<Ix1>().ok().unwrap();
            a_grad = Some(x2.dot(&y1).into_dyn());
        } else if dy == 2 {
            let y2 = y.clone().into_dimensionality::<Ix2>().ok().unwrap();
            a_grad = Some(x2.dot(&y2).into_dyn());
        } else {
            panic!("Illegal sized 'a' array");
        }
    } else {
        panic!("Illegal sized 'x' array");
    }

    return a_grad.unwrap();
}

pub fn abs(a: ArrayD<f64>) -> ArrayD<f64> {
    let mut b = a.clone();
    for i in b.iter_mut() {
        *i = i.abs();
    }
    b
}

#[derive(Clone)]
pub struct ShapeError {
    // we want to be able to change this representation later
    repr: ErrorKind,
}

impl ShapeError {
    /// Return the `ErrorKind` of this error.
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        self.repr
    }

    /// Create a new `ShapeError`
    pub fn from_kind(error: ErrorKind) -> Self {
        from_kind(error)
    }
}

#[inline(always)]
pub fn from_kind(k: ErrorKind) -> ShapeError {
    ShapeError { repr: k }
}

#[derive(Copy, Clone, Debug)]
pub enum ErrorKind {
    /// incompatible shape
    IncompatibleShape = 1,
    /// incompatible memory layout
    IncompatibleLayout,
    /// the shape does not fit inside type limits
    RangeLimited,
    /// out of bounds indexing
    OutOfBounds,
    /// aliasing array elements
    Unsupported,
    #[doc(hidden)]
    __Incomplete,
}

pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError> {
    let size_nonzero = dim
        .slice()
        .iter()
        .filter(|&&d| d != 0)
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ShapeError {
            repr: ErrorKind::OutOfBounds,
        })?;
    if size_nonzero > ::std::isize::MAX as usize {
        Err(ShapeError {
            repr: ErrorKind::OutOfBounds,
        })
    } else {
        Ok(dim.size())
    }
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ShapeError/{:?}: {}", self.kind(), self.description())
    }
}

impl fmt::Debug for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ShapeError/{:?}: {}", self.kind(), self.description())
    }
}

impl Error for ShapeError {
    fn description(&self) -> &str {
        match self.kind() {
            ErrorKind::IncompatibleShape => "incompatible shapes",
            ErrorKind::IncompatibleLayout => "incompatible memory layout",
            ErrorKind::RangeLimited => "the shape does not fit in type limits",
            ErrorKind::OutOfBounds => "out of bounds indexing",
            ErrorKind::Unsupported => "unsupported operation",
            ErrorKind::__Incomplete => "this error variant is not in use",
        }
    }
}

pub fn sum_axis(a: ArrayD<f64>, axis: Axis, keep_dim: bool) -> ArrayD<f64> {
    let n = a.len_of(axis);
    let mut res = Array::zeros(a.raw_dim());
    res = res.remove_axis(axis);
    let stride = a.strides()[axis.index()];

    let len: usize = res.dim().size();

    if a.ndim() == 2 && stride == 1 {
        // contiguous along the axis we are summing
        let ax = axis.index();
        for (i, elt) in enumerate(&mut res) {
            if i < len {
                *elt = a.index_axis(Axis(1 - ax), i).sum();
            }
        }
    } else {
        for i in 0..n {
            let view = a.index_axis(axis, i);
            res = res + &view;
        }
    }
    if keep_dim && axis.index() != 0 {
        let shared = res.to_shared();
        let mut new_dim = a.dim().clone();
        new_dim[axis.index()] = 1;
        let shaped = shared.reshape(new_dim);
        shaped.into_dyn().into_owned()
    } else {
        res
    }
}
