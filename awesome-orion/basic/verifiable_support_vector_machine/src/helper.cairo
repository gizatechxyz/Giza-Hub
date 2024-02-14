use debug::PrintTrait;
use traits::TryInto;
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{
    Tensor, TensorTrait, FP16x16Tensor, FP16x16TensorAdd, FP16x16TensorMul, FP16x16TensorSub,
    FP16x16TensorDiv
};
use orion::numbers::{FixedTrait, FP16x16, FP16x16Impl};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    HALF, ONE, FP16x16Mul, FP16x16Div, FP16x16Print, FP16x16IntoI32, FP16x16PartialOrd,
    FP16x16PartialEq
};

// Calculates the machine learning model's loss.
fn calculate_loss(
    w: @Tensor<FP16x16>,
    x_train: @Tensor<FP16x16>,
    y_train: @Tensor<FP16x16>,
    c: @Tensor<FP16x16>,
    one_tensor: @Tensor<FP16x16>,
    half_tensor: @Tensor<FP16x16>,
    y_train_len: u32
) -> FP16x16 {
    let tensor_size = FixedTrait::new_unscaled(y_train_len, false);

    let pre_cumsum = *one_tensor - *y_train * x_train.matmul(w);
    let cumsum = pre_cumsum.cumsum(0, Option::None(()), Option::None(()));
    let sum = cumsum.data[pre_cumsum.data.len() - 1];
    let mean = FP16x16Div::div(*sum, tensor_size);

    let mean_tensor = TensorTrait::new(
        shape: array![1].span(), data: array![mean].span(),
    );

    let regularization_term = *half_tensor * (w.matmul(w));
    let loss_tensor = mean_tensor + *c * regularization_term;

    loss_tensor.at(array![0].span())
}

// Calculates the gradient for the machine learning model
fn calculate_gradient(
    w: @Tensor<FP16x16>,
    x_train: @Tensor<FP16x16>,
    y_train: @Tensor<FP16x16>,
    c: Tensor<FP16x16>,
    one_tensor: @Tensor<FP16x16>,
    neg_one_tensor: @Tensor<FP16x16>,
    y_train_len: u32
) -> Tensor<FP16x16> {
    let tensor_size = TensorTrait::new(
        shape: array![1].span(),
        data: array![FixedTrait::new_unscaled(y_train_len, false)].span(),
    );

    let mask = (*y_train * x_train.matmul(w));
    let mask = less(@mask, one_tensor);

    let gradient = (((mask * *y_train).matmul(x_train) / tensor_size) * *neg_one_tensor) + (c * *w);

    gradient
}

// Calculates the accuracy of the machine learning model's predictions.
fn accuracy(y: @Tensor<FP16x16>, z: @Tensor<FP16x16>) -> FP16x16 {
    let (mut left, mut right) = (y, z);

    let mut right_data = *right.data;
    let mut left_data = *left.data;
    let mut counter = 0;

    loop {
        match right_data.pop_front() {
            Option::Some(item) => {
                let right_current_index = item;
                let left_current_index = left_data.pop_front();
                let (y_value, z_value) = (left_current_index.unwrap(), right_current_index);

                if *y_value == *z_value {
                    counter += 1;
                };
            },
            Option::None(_) => {
                break;
            }
        };
    };

    (FixedTrait::new_unscaled(counter, false) / FixedTrait::new_unscaled((*y.data).len(), false))
        * FixedTrait::new_unscaled(100, false)
}

// Returns the truth value of (x < y) element-wise.
fn less(y: @Tensor<FP16x16>, z: @Tensor<FP16x16>) -> Tensor<FP16x16> {
    let mut data_result = ArrayTrait::<FP16x16>::new();
    let mut data_result2 = ArrayTrait::<FP16x16>::new();
    let (mut smaller, mut bigger, retains_input_order) = if (*y.data).len() < (*z.data).len() {
        (y, z, true)
    } else {
        (z, y, false)
    };

    let mut bigger_data = *bigger.data;
    let mut smaller_data = *smaller.data;
    let mut smaller_index = 0;

    loop {
        match bigger_data.pop_front() {
            Option::Some(item) => {
                let bigger_current_index = item;
                let smaller_current_index = smaller_data[smaller_index];

                let (y_value, z_value) = if retains_input_order {
                    (smaller_current_index, bigger_current_index)
                } else {
                    (bigger_current_index, smaller_current_index)
                };

                if *y_value < *z_value {
                    data_result.append(FixedTrait::ONE());
                } else {
                    data_result.append(FixedTrait::ZERO());
                };

                smaller_index = (1 + smaller_index) % smaller_data.len();
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<FP16x16>::new(*bigger.shape, data_result.span());
}


// Returns an element-wise indication of the sign of a number.
fn sign(z: @Tensor<FP16x16>) -> Tensor<FP16x16> {
    let mut data_result = ArrayTrait::<FP16x16>::new();
    let mut z_data = *z.data;

    loop {
        match z_data.pop_front() {
            Option::Some(item) => {
                let result = if *item.sign {
                    FixedTrait::new(ONE, true)
                } else {
                    FixedTrait::new(ONE, false)
                };
                data_result.append(result);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    TensorTrait::<FP16x16>::new(*z.shape, data_result.span())
}

// Returns predictions using the machine learning model.
fn pred(x: @Tensor<FP16x16>, w: @Tensor<FP16x16>) -> Tensor<FP16x16> {
    sign(@(x.matmul(w)))
}
