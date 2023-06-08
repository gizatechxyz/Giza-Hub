use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 1300, sign: true });
    data.append(i32 { mag: 7644, sign: false });
    data.append(i32 { mag: 472, sign: true });
    data.append(i32 { mag: 3601, sign: false });
    data.append(i32 { mag: 5538, sign: false });
    data.append(i32 { mag: 5476, sign: false });
    data.append(i32 { mag: 3879, sign: false });
    data.append(i32 { mag: 3268, sign: true });
    data.append(i32 { mag: 1979, sign: false });
    data.append(i32 { mag: 1435, sign: false });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
