use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{core::{Tensor, TensorTrait}};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FixedTrait};

fn evec_sort() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::new();
    shape.append(3);
    shape.append(3);

    // evec Original
    // [[ 36422 -38024 -38467]
    // [ 53777  30012  21440]
    // [  5195 -43789  48143]]

    // evec Sorted
    // [[-38024  36422 -38467]
    // [ 30012  53777  21440]
    // [-43789   5195  48143]]

    let mut data = ArrayTrait::new();

    data.append(FixedTrait::new(38024, true));
    data.append(FixedTrait::new(36422, false));
    data.append(FixedTrait::new(38467, true));

    data.append(FixedTrait::new(30012, false));
    data.append(FixedTrait::new(53777, false));
    data.append(FixedTrait::new(21440, false));

    data.append(FixedTrait::new(43789, true));
    data.append(FixedTrait::new(5195, false));
    data.append(FixedTrait::new(48143, false));

    let tensor = TensorTrait::<FP16x16>::new(shape.span(), data.span());
    return tensor;
}
