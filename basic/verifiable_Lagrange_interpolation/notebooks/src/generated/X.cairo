use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{core::{Tensor, TensorTrait}};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FixedTrait};

fn X() -> Tensor<FP16x16>{

let mut shape = ArrayTrait::new();
shape.append(11);
let mut data = ArrayTrait::new();
data.append(FixedTrait::new(327680, false));
data.append(FixedTrait::new(311642, false));
data.append(FixedTrait::new(265098, false));
data.append(FixedTrait::new(192605, false));
data.append(FixedTrait::new(101258, false));
data.append(FixedTrait::new(0, false));
data.append(FixedTrait::new(101258, true));
data.append(FixedTrait::new(192605, true));
data.append(FixedTrait::new(265098, true));
data.append(FixedTrait::new(311642, true));
data.append(FixedTrait::new(327680, true));
let tensor = TensorTrait::<FP16x16>::new(shape.span(), data.span());
return tensor;
}