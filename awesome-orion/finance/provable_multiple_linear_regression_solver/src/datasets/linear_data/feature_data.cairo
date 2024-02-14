use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn x_feature_data() -> Tensor<FP16x16> {
    let tensor = TensorTrait::<
        FP16x16
    >::new(
        shape: array![50].span(),
        data: array![
            FixedTrait::new(19671, false),
            FixedTrait::new(23085, true),
            FixedTrait::new(74876, true),
            FixedTrait::new(22894, true),
            FixedTrait::new(13690, true),
            FixedTrait::new(38444, false),
            FixedTrait::new(54983, false),
            FixedTrait::new(61020, false),
            FixedTrait::new(18716, false),
            FixedTrait::new(58008, false),
            FixedTrait::new(49440, true),
            FixedTrait::new(82107, false),
            FixedTrait::new(33615, false),
            FixedTrait::new(19535, true),
            FixedTrait::new(32015, false),
            FixedTrait::new(4952, true),
            FixedTrait::new(74162, false),
            FixedTrait::new(99602, false),
            FixedTrait::new(143233, false),
            FixedTrait::new(91520, true),
            FixedTrait::new(94641, true),
            FixedTrait::new(33060, true),
            FixedTrait::new(10488, false),
            FixedTrait::new(57420, false),
            FixedTrait::new(20685, false),
            FixedTrait::new(132526, true),
            FixedTrait::new(20067, true),
            FixedTrait::new(54262, false),
            FixedTrait::new(15079, false),
            FixedTrait::new(49939, false),
            FixedTrait::new(14570, true),
            FixedTrait::new(13156, true),
            FixedTrait::new(12226, false),
            FixedTrait::new(26873, false),
            FixedTrait::new(12995, false),
            FixedTrait::new(7799, false),
            FixedTrait::new(43952, true),
            FixedTrait::new(24744, false),
            FixedTrait::new(7983, false),
            FixedTrait::new(74021, false),
            FixedTrait::new(78572, false),
            FixedTrait::new(12134, false),
            FixedTrait::new(24594, true),
            FixedTrait::new(41859, true),
            FixedTrait::new(27754, false),
            FixedTrait::new(5068, false),
            FixedTrait::new(22534, true),
            FixedTrait::new(2857, false),
            FixedTrait::new(40632, true),
            FixedTrait::new(45746, false),
        ]
            .span()
    );

    return tensor;
}
