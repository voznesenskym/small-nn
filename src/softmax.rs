use std::ops::Div;

use ndarray::{array, s, Array, ArrayD, Axis, Slice};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

use crate::utility::sum_axis;

pub struct SoftmaxCrossentropyWithLogits {
    x: ArrayD<f64>,
    y: ArrayD<f64>,
    softmax: ArrayD<f64>,
}

impl SoftmaxCrossentropyWithLogits {
    pub fn new() -> SoftmaxCrossentropyWithLogits {
        let x = Array::random((1, 1), Uniform::new(10., 100.)).into_dyn();
        let y = Array::random((1, 1), Uniform::new(10., 100.)).into_dyn();

        let softmax = Array::random((1, 1), Uniform::new(10., 100.)).into_dyn();

        SoftmaxCrossentropyWithLogits { x, y, softmax }
    }

    pub fn forward(&mut self, x: ArrayD<f64>, y: ArrayD<f64>) -> ArrayD<f64> {
        self.x = x;
        self.y = y;

        for i in self.x.iter_mut() {
            *i = f64::exp(*i);
        }
        //    println!("X muted is {}", self.x);
        let shape = self.x.shape();
        let num_axis = shape.len();
        self.softmax = self.x.clone() / sum_axis(self.x.clone(), Axis(num_axis - 1), true);
        //        println!("softmax is {}", self.softmax);

        let interior_arange = std::ops::Range {
            start: 0,
            end: shape[0],
        };
        let slice = Slice::from(interior_arange);

        let y = self.y[0] as usize;
        let mut logits = self.softmax.slice(s![slice, y]).into_owned();
        //    println!("Logits: {}", logits);
        for i in logits.iter_mut() {
            *i = -f64::ln(*i);
        }
        //    println!("Logits mut: {}", logits);
        let loss = logits.sum() / self.x.shape()[0] as f64;
        return array![loss].into_dyn();
    }

    pub fn backward(&mut self) -> ArrayD<f64> {
        let batch = self.x.shape()[0];
        let mut grad = self.softmax.clone();
        let interior_arange = std::ops::Range {
            start: 0,
            end: batch,
        };
        let slice = Slice::from(interior_arange);
        //    println!("Grad: {}", grad);
        //        println!("slice: {:?}", slice);
        let mut grad_slice = grad.slice_mut(s![slice, self.y[0] as usize]);
        for i in grad_slice.iter_mut() {
            *i = *i - 1.0;
        }
        grad = grad.div(batch as f64);
        return grad;
    }
}
