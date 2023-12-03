use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{core::{Tensor, TensorTrait}};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FixedTrait};

fn evalu_sort() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::new();
    shape.append(3);
    let mut data = ArrayTrait::new();

    // evalu Original [ 52513 137534   5393]
    // evalu Sorted   [137534  52513   5393]

    data.append(FixedTrait::new(137534, false));
    data.append(FixedTrait::new(52513, false));
    data.append(FixedTrait::new(5393, false));

    let tensor = TensorTrait::<FP16x16>::new(shape.span(), data.span());
    return tensor;
}
