use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};


fn user_input_boston_housing() -> Tensor<FP16x16> {
    let tensor = TensorTrait::<
        FP16x16
    >::new(
        shape: array![11].span(),
        data: array![
            FixedTrait::new(26719, false),
            FixedTrait::new(0, false),
            FixedTrait::new(406323, false),
            FixedTrait::new(65536, false),
            FixedTrait::new(33226, false),
            FixedTrait::new(403963, false),
            FixedTrait::new(5983436, false),
            FixedTrait::new(199753, false),
            FixedTrait::new(524288, false),
            FixedTrait::new(20119552, false),
            FixedTrait::new(1140326, false),
        ]
            .span()
    );
    return tensor;
}
