use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{core::{Tensor, TensorTrait}};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FixedTrait};

fn y() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::new();
    shape.append(105);
    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(131072, false));
    data.append(FixedTrait::new(131072, false));
    data.append(FixedTrait::new(131072, false));
    data.append(FixedTrait::new(131072, false));
    data.append(FixedTrait::new(131072, false));
    let tensor = TensorTrait::<FP16x16>::new(shape.span(), data.span());
    return tensor;
}
