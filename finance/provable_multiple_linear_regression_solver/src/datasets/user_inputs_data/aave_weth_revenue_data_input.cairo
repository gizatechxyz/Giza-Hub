use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn aave_weth_revenue_data_input() -> Tensor<FP16x16> {
    let tensor = TensorTrait::<
        FP16x16
    >::new(
        shape: array![7, 9].span(),
        data: array![
            FixedTrait::new(160, false),
            FixedTrait::new(786432, false),
            FixedTrait::new(1048576, false),
            FixedTrait::new(16973824, false),
            FixedTrait::new(4128, false),
            FixedTrait::new(10354688, false),
            FixedTrait::new(1952972, false),
            FixedTrait::new(5852364, false),
            FixedTrait::new(198574079, false),
            FixedTrait::new(185, false),
            FixedTrait::new(681574, false),
            FixedTrait::new(1048576, false),
            FixedTrait::new(17170432, false),
            FixedTrait::new(4128, false),
            FixedTrait::new(10420224, false),
            FixedTrait::new(1959526, false),
            FixedTrait::new(5891686, false),
            FixedTrait::new(207093760, false),
            FixedTrait::new(211, false),
            FixedTrait::new(688128, false),
            FixedTrait::new(1055129, false),
            FixedTrait::new(17301504, false),
            FixedTrait::new(4128, false),
            FixedTrait::new(10420224, false),
            FixedTrait::new(1952972, false),
            FixedTrait::new(5963776, false),
            FixedTrait::new(206438400, false),
            FixedTrait::new(236, false),
            FixedTrait::new(707788, false),
            FixedTrait::new(1055129, false),
            FixedTrait::new(17367040, false),
            FixedTrait::new(4128, false),
            FixedTrait::new(10420224, false),
            FixedTrait::new(1907097, false),
            FixedTrait::new(6035865, false),
            FixedTrait::new(203161600, false),
            FixedTrait::new(261, false),
            FixedTrait::new(792985, false),
            FixedTrait::new(1061683, false),
            FixedTrait::new(17432576, false),
            FixedTrait::new(4128, false),
            FixedTrait::new(10420224, false),
            FixedTrait::new(1880883, false),
            FixedTrait::new(6134169, false),
            FixedTrait::new(195952639, false),
            FixedTrait::new(285, false),
            FixedTrait::new(792985, false),
            FixedTrait::new(1061683, false),
            FixedTrait::new(17432576, false),
            FixedTrait::new(4128, false),
            FixedTrait::new(10420224, false),
            FixedTrait::new(1880883, false),
            FixedTrait::new(6153830, false),
            FixedTrait::new(195952639, false),
            FixedTrait::new(308, false),
            FixedTrait::new(792985, false),
            FixedTrait::new(1061683, false),
            FixedTrait::new(17498112, false),
            FixedTrait::new(4128, false),
            FixedTrait::new(10420224, false),
            FixedTrait::new(1887436, false),
            FixedTrait::new(6180044, false),
            FixedTrait::new(196607999, false),
        ]
            .span()
    );

    return tensor;
}
