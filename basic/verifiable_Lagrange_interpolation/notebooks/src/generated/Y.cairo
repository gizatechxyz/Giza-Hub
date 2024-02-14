use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{core::{Tensor, TensorTrait}};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16, FixedTrait};

fn Y() -> Tensor<FP16x16>{

let mut shape = ArrayTrait::new();
shape.append(11);
let mut data = ArrayTrait::new();
data.append(FixedTrait::new(2520, false));
data.append(FixedTrait::new(2775, false));
data.append(FixedTrait::new(3774, false));
data.append(FixedTrait::new(6800, false));
data.append(FixedTrait::new(19347, false));
data.append(FixedTrait::new(65536, false));
data.append(FixedTrait::new(19347, false));
data.append(FixedTrait::new(6800, false));
data.append(FixedTrait::new(3774, false));
data.append(FixedTrait::new(2775, false));
data.append(FixedTrait::new(2520, false));
let tensor = TensorTrait::<FP16x16>::new(shape.span(), data.span());
return tensor;
}