use ndarray::ArrayD;

use crate::layer::Layer;

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Model {
        Model { layers }
    }

    pub fn forward(&mut self, x: ArrayD<f64>) -> ArrayD<f64> {
        let mut curr_x = x;

        for layer in &mut self.layers {
            curr_x = layer.forward(curr_x.clone());
        }
        curr_x
    }

    pub fn backward(&mut self, grad: ArrayD<f64>) -> ArrayD<f64> {
        let mut curr_grad = grad;

        for layer in self.layers.iter_mut().rev() {
            curr_grad = layer.backward(curr_grad.clone());
        }
        curr_grad
    }
}
