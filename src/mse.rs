use ndarray::{Array, ArrayD};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

pub struct MSE {
    x: ArrayD<f64>,
    y: ArrayD<f64>,
}

impl MSE {
    pub fn new() -> MSE {
        let x = Array::random((1, 1), Uniform::new(10., 100.)).into_dyn();
        let y = Array::random((1, 1), Uniform::new(10., 100.)).into_dyn();

        MSE { x, y }
    }

    pub fn forward(&mut self, x: ArrayD<f64>, y: ArrayD<f64>) -> ArrayD<f64> {
        self.x = x;
        self.y = y;

        let interm = (self.x.clone() - self.y.clone()) * (self.x.clone() - self.y.clone());

        return (interm) / (self.x.shape()[0] * 2) as f64;
    }

    pub fn backward(&mut self) -> ArrayD<f64> {
        (self.x.clone() - self.y.clone()) / self.x.clone().shape()[0] as f64
    }
}
