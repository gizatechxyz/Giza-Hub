use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq};
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn y_label_data() -> Tensor<FP16x16> {
    let tensor = TensorTrait::<
        FP16x16
    >::new(
        shape: array![50].span(),
        data: array![
            FixedTrait::new(6817428, false),
            FixedTrait::new(6328827, false),
            FixedTrait::new(5919977, false),
            FixedTrait::new(6300125, false),
            FixedTrait::new(6500794, false),
            FixedTrait::new(6710325, false),
            FixedTrait::new(7107816, false),
            FixedTrait::new(6991879, false),
            FixedTrait::new(6724238, false),
            FixedTrait::new(7001326, false),
            FixedTrait::new(6253898, false),
            FixedTrait::new(7075450, false),
            FixedTrait::new(6801393, false),
            FixedTrait::new(6372144, false),
            FixedTrait::new(6884026, false),
            FixedTrait::new(6441896, false),
            FixedTrait::new(7135599, false),
            FixedTrait::new(7292890, false),
            FixedTrait::new(7702237, false),
            FixedTrait::new(5859629, false),
            FixedTrait::new(5724338, false),
            FixedTrait::new(6364135, false),
            FixedTrait::new(6696592, false),
            FixedTrait::new(7045896, false),
            FixedTrait::new(6778122, false),
            FixedTrait::new(5448575, false),
            FixedTrait::new(6385007, false),
            FixedTrait::new(6926370, false),
            FixedTrait::new(6656679, false),
            FixedTrait::new(6987870, false),
            FixedTrait::new(6391707, false),
            FixedTrait::new(6422343, false),
            FixedTrait::new(6606377, false),
            FixedTrait::new(6713193, false),
            FixedTrait::new(6613575, false),
            FixedTrait::new(6615164, false),
            FixedTrait::new(6128755, false),
            FixedTrait::new(6766914, false),
            FixedTrait::new(6726246, false),
            FixedTrait::new(7194405, false),
            FixedTrait::new(7169606, false),
            FixedTrait::new(6592503, false),
            FixedTrait::new(6307876, false),
            FixedTrait::new(6329638, false),
            FixedTrait::new(6778962, false),
            FixedTrait::new(6552402, false),
            FixedTrait::new(6385833, false),
            FixedTrait::new(6714099, false),
            FixedTrait::new(6236415, false),
            FixedTrait::new(6960018, false),
        ]
            .span()
    );

    return tensor;
}
