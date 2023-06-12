use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::numbers::fixed_point::core::FixedImpl;
use orion::numbers::signed_integer::i32::i32;

fn fc2_bias() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 851, sign: true });
    data.append(i32 { mag: 336, sign: false });
    data.append(i32 { mag: 451, sign: false });
    data.append(i32 { mag: 406, sign: true });
    data.append(i32 { mag: 297, sign: false });
    data.append(i32 { mag: 912, sign: false });
    data.append(i32 { mag: 617, sign: true });
    data.append(i32 { mag: 70, sign: false });
    data.append(i32 { mag: 1225, sign: true });
    data.append(i32 { mag: 486, sign: false });
let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) }; 
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}
