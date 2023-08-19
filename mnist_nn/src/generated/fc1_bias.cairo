use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc1_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 4694, sign: true });
    data.append(i32 { mag: 1163, sign: true });
    data.append(i32 { mag: 4970, sign: false });
    data.append(i32 { mag: 6878, sign: false });
    data.append(i32 { mag: 613, sign: false });
    data.append(i32 { mag: 4506, sign: false });
    data.append(i32 { mag: 3042, sign: false });
    data.append(i32 { mag: 5345, sign: false });
    data.append(i32 { mag: 836, sign: true });
    data.append(i32 { mag: 516, sign: false });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
