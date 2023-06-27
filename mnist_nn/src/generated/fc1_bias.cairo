use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 6867, sign: false });
    data.append(i32 { mag: 1374, sign: true });
    data.append(i32 { mag: 1248, sign: true });
    data.append(i32 { mag: 845, sign: true });
    data.append(i32 { mag: 1639, sign: true });
    data.append(i32 { mag: 6442, sign: false });
    data.append(i32 { mag: 11312, sign: false });
    data.append(i32 { mag: 8018, sign: false });
    data.append(i32 { mag: 5604, sign: true });
    data.append(i32 { mag: 2109, sign: true });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
