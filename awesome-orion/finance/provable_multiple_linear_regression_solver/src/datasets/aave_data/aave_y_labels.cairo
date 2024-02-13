use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn aave_y_labels() -> Tensor<FP16x16> {
    let tensor = TensorTrait::<
        FP16x16
    >::new(
        shape: array![24].span(),
        data: array![
            FixedTrait::new(5072486, false),
            FixedTrait::new(5072486, false),
            FixedTrait::new(5079040, false),
            FixedTrait::new(5085593, false),
            FixedTrait::new(5111808, false),
            FixedTrait::new(5157683, false),
            FixedTrait::new(5203558, false),
            FixedTrait::new(5360844, false),
            FixedTrait::new(5367398, false),
            FixedTrait::new(5367398, false),
            FixedTrait::new(5452595, false),
            FixedTrait::new(5485363, false),
            FixedTrait::new(5505024, false),
            FixedTrait::new(5583667, false),
            FixedTrait::new(5681971, false),
            FixedTrait::new(5754060, false),
            FixedTrait::new(5780275, false),
            FixedTrait::new(5852364, false),
            FixedTrait::new(5891686, false),
            FixedTrait::new(5963776, false),
            FixedTrait::new(6035865, false),
            FixedTrait::new(6134169, false),
            FixedTrait::new(6153830, false),
            FixedTrait::new(6180044, false),
        ]
            .span()
    );

    return tensor;
}
