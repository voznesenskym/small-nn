use std::ops::Sub;

use ndarray::{Array, ArrayD, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::utility::dot;

pub trait Layer {
    fn forward(&mut self, x: ArrayD<f64>) -> ArrayD<f64>;
    fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64>;
}

pub struct Linear {
    a: ArrayD<f64>,
    b: ArrayD<f64>,
    x: Option<ArrayD<f64>>,
    lr: f64,
}

impl Linear {
    pub fn new(input: usize, output: usize, lr: f64) -> Linear {
        let a = 2.0 as f64 * Array::random((input, output), Uniform::new(0., 1.)).into_dyn()
            - 1.0 as f64;
        let b = 2.0 as f64 * Array::random(output, Uniform::new(0., 1.)).into_dyn() - 1.0 as f64;
        Linear { a, b, lr, x: None }
    }
}

impl Layer for Linear {
    fn forward(&mut self, x: ArrayD<f64>) -> ArrayD<f64> {
        self.x = Some(x);
        let x = self.x.clone().unwrap();
        let a = &self.a;
        let res = dot(&x, a);

        return res + self.b.clone();
    }

    fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
        let x = self.x.as_ref().unwrap();
        let b_grad = grad.mean_axis(Axis(0)).unwrap() * x.shape()[0] as f64;

        let a_grad = dot(&x.t().into_owned(), &grad);

        let a = &self.a;
        let grad_input = dot(&grad, &a.t().into_owned());

        self.a = self.a.clone().sub(a_grad * self.lr);
        self.b = self.b.clone().sub(b_grad * self.lr);

        return grad_input;
    }
}

pub struct ReLu {
    x: Option<ArrayD<f64>>,
}

impl Layer for ReLu {
    fn forward(&mut self, x: ArrayD<f64>) -> ArrayD<f64> {
        self.x = Some(x);

        let mut xc = self.x.clone().unwrap();
        for i in xc.iter_mut() {
            if i < &mut 0.0 {
                *i = 0.0
            }
        }

        xc
    }

    fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
        let mut xc = self.x.clone().unwrap();
        for i in xc.iter_mut() {
            if i < &mut 0.0 {
                *i = 0.0
            } else {
                *i = 1.0
            }
        }
        return xc * grad;
    }
}

impl ReLu {
    pub fn new() -> ReLu {
        ReLu { x: None }
    }
}
