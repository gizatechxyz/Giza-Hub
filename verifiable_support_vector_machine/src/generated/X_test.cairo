use array::ArrayTrait;
use orion::operators::tensor::{
    core::{Tensor, TensorTrait, ExtraParams}, implementations::impl_tensor_fp::Tensor_fp
};
use orion::numbers::fixed_point::{
    core::{FixedTrait, FixedType, FixedImpl}, implementations::fp16x16::core::FP16x16Impl
};

fn X_test() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::new();
    shape.append(50);
    shape.append(3);
    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new(87946, false));
    data.append(FixedTrait::new(38900, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(34695, false));
    data.append(FixedTrait::new(249556, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(159279, false));
    data.append(FixedTrait::new(4166, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(83319, false));
    data.append(FixedTrait::new(124029, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(154108, false));
    data.append(FixedTrait::new(54263, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(81434, false));
    data.append(FixedTrait::new(295173, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(101339, false));
    data.append(FixedTrait::new(276101, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(51730, false));
    data.append(FixedTrait::new(284261, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(130047, false));
    data.append(FixedTrait::new(32083, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(112357, false));
    data.append(FixedTrait::new(329332, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(103102, false));
    data.append(FixedTrait::new(31715, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(111245, false));
    data.append(FixedTrait::new(56762, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(77993, false));
    data.append(FixedTrait::new(309836, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(180005, false));
    data.append(FixedTrait::new(101281, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(62872, false));
    data.append(FixedTrait::new(298895, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(36406, true));
    data.append(FixedTrait::new(307754, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(150385, false));
    data.append(FixedTrait::new(50193, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(106162, false));
    data.append(FixedTrait::new(4433, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(79801, false));
    data.append(FixedTrait::new(255125, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(95363, false));
    data.append(FixedTrait::new(1913, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(121, true));
    data.append(FixedTrait::new(300250, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(14718, false));
    data.append(FixedTrait::new(312625, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(89573, false));
    data.append(FixedTrait::new(41613, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(30116, false));
    data.append(FixedTrait::new(357159, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(159865, false));
    data.append(FixedTrait::new(4752, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(61297, false));
    data.append(FixedTrait::new(349424, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(116464, false));
    data.append(FixedTrait::new(77761, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(70534, false));
    data.append(FixedTrait::new(307023, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(155183, false));
    data.append(FixedTrait::new(36188, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(133737, false));
    data.append(FixedTrait::new(29808, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(145701, false));
    data.append(FixedTrait::new(54969, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(105613, false));
    data.append(FixedTrait::new(119503, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(138536, false));
    data.append(FixedTrait::new(81751, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(89231, false));
    data.append(FixedTrait::new(89547, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(37538, false));
    data.append(FixedTrait::new(267914, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(211603, false));
    data.append(FixedTrait::new(74168, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(157689, false));
    data.append(FixedTrait::new(319191, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(145953, false));
    data.append(FixedTrait::new(82769, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(69644, false));
    data.append(FixedTrait::new(339237, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(108783, false));
    data.append(FixedTrait::new(233497, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(120774, false));
    data.append(FixedTrait::new(4764, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(38978, false));
    data.append(FixedTrait::new(308651, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(122543, false));
    data.append(FixedTrait::new(7073, true));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(111206, false));
    data.append(FixedTrait::new(49473, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(130347, false));
    data.append(FixedTrait::new(98944, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(161886, false));
    data.append(FixedTrait::new(86147, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(86653, false));
    data.append(FixedTrait::new(273862, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(86507, false));
    data.append(FixedTrait::new(92030, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(124252, false));
    data.append(FixedTrait::new(339830, false));
    data.append(FixedTrait::new(65536, false));
    data.append(FixedTrait::new(113347, false));
    data.append(FixedTrait::new(75189, false));
    data.append(FixedTrait::new(65536, false));
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let tensor = TensorTrait::<FixedType>::new(shape.span(), data.span(), Option::Some(extra));
    return tensor;
}