pub mod layer;
mod model;
mod mse;
mod softmax;
mod utility;
use std::time::SystemTime;

use crate::softmax::SoftmaxCrossentropyWithLogits;
use crate::utility::argmax;
use layer::{Linear, ReLu};
use mnist::*;
use model::Model;
use mse::MSE;
use ndarray::prelude::*;
use ndarray::{array, ArrayD};
use ndarray_rand::rand::{self, Rng};

fn loss_y_arr(x1: f64, x2: f64, x3: f64) -> ArrayD<f64> {
    return array![(x1 * 2.0) + (x2 * 3.0) + (x3 * 4.0) + 5.0].into_dyn();
}

fn mse_test() {
    let lr = 0.0001;
    let mut model = Model::new(vec![
        Box::new(Linear::new(3, 15, lr)),
        Box::new(ReLu::new()),
        Box::new(Linear::new(15, 1, lr)),
    ]);

    let mut loss = MSE::new();
    for i in 0..20000 {
        let x1 = rand::thread_rng().gen_range(0.0..1.0) * 30.0;

        let x2 = rand::thread_rng().gen_range(0.0..1.0) * 20.0;

        let x3 = rand::thread_rng().gen_range(0.0..1.0) * 11.0;

        let yarr = array![[x1, x2, x3]];

        let y = model.forward(yarr.into_dyn());

        let error = loss.forward(y, loss_y_arr(x1, x2, x3));

        let mut yb = loss.backward();
        yb = model.backward(yb);

        if i % 1000 == 0 {
            println!("Err: {}", error);
            println!("Val target: {}", loss_y_arr(1.0, 2.0, 3.0));
            println!("Res: {}", model.forward(array![[1.0, 2.0, 3.0]].into_dyn()));
            println!(
                "Loss: {}",
                loss.forward(
                    loss_y_arr(1.0, 2.0, 3.0),
                    model.forward(array![[1.0, 2.0, 3.0]].into_dyn())
                )
            );
            println!("---------")
        }
    }
}

fn mnist_test() {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);

    let train_data_reshaped = train_data.to_shape((50_000, 1, 784)).unwrap();
    println!(
        "{:#.1?}\n",
        train_data_reshaped.slice(s![image_num, .., ..])
    );

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f64> = Array2::from_shape_vec((50_000, 1), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    println!(
        "The first digit is a {:?}",
        train_labels.slice(s![image_num, ..])
    );

    let test_data = Array3::from_shape_vec((10_000, 28, 28), tst_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.);

    let test_data_reshaped = test_data.to_shape((10_000, 1, 784)).unwrap();

    let test_labels: Array2<f64> = Array2::from_shape_vec((10_000, 1), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    let lr = 0.0001;

    // 784 is 28 * 28
    let mut model = Model::new(vec![
        Box::new(Linear::new(784, 100, lr)),
        Box::new(ReLu::new()),
        Box::new(Linear::new(100, 200, lr)),
        Box::new(ReLu::new()),
        Box::new(Linear::new(200, 10, lr)),
    ]);
    let mut loss = SoftmaxCrossentropyWithLogits::new();
    let mut batch_avg_loss = 0.0;
    let mut whole_avg_loss = 0.0;

    let mut rng = rand::thread_rng();

    let mut n = 1.0;
    for _x in 0..7 {
        let start = SystemTime::now();

        for (i, _) in train_data_reshaped.iter().enumerate() {
            if i == 50_000 {
                // Bug with slice at max?
                break;
            }
            let train = train_data_reshaped.slice(s![i, .., ..]);
            let label = train_labels.slice(s![i, ..]);

            let res = model.forward(train.into_owned().into_dyn());
            let err = loss.forward(res, label.into_owned().into_dyn());
            batch_avg_loss += err.mean().unwrap();
            whole_avg_loss += err.mean().unwrap();

            let grad = loss.backward();
            model.backward(grad);

            if i % 1000 == 0 {
                batch_avg_loss = batch_avg_loss / 1.0;
                let current_loss = whole_avg_loss / n;
                let end = SystemTime::now();
                let span = end.duration_since(start).expect("Time").as_secs();
                println!("Batch loss: {} at [{}]", batch_avg_loss, span);
                println!("Current loss: {} at [{}]", current_loss, span);

                batch_avg_loss = 0.0;

                let mut correct = 0.0;
                let mut total = 0.0;
                for r in 0..10 {
                    let mut random_index: i32 = rng.gen();
                    random_index = random_index % 9999;
                    let random_test_data = test_data_reshaped.slice(s![random_index, .., ..]);
                    let random_test_label = test_labels.slice(s![random_index, ..]);

                    let data = random_test_data.into_owned().into_dyn();
                    let mut res = model.forward(data);
                    let label = random_test_label.into_owned().into_dyn();
                    res = res.remove_axis(Axis(0));
                    let chosen_index = argmax(res) as i64;
                    let chosen_label = label[0] as i64;
                    println!("chosen_index: {} | correct_index: {}", chosen_index, chosen_label);

                    if chosen_index == chosen_label {
                        correct += 1.0;
                    }
                    total += 1.0;
                }
                println!("Score is: {}", correct / total);
            }
            n += 1.0;
        }
    }
}

fn main() {
    mnist_test();
}
