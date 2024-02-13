use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn tensor() -> Tensor<FP16x16> {
    Tensor {
        shape: array![18,].span(),
        data: array![
            FP16x16 { mag: 77099, sign: true },
            FP16x16 { mag: 140625, sign: true },
            FP16x16 { mag: 59258, sign: true },
            FP16x16 { mag: 40408, sign: false },
            FP16x16 { mag: 66634, sign: true },
            FP16x16 { mag: 27092, sign: false },
            FP16x16 { mag: 3196, sign: true },
            FP16x16 { mag: 392, sign: true },
            FP16x16 { mag: 81839, sign: true },
            FP16x16 { mag: 39890, sign: true },
            FP16x16 { mag: 16196, sign: false },
            FP16x16 { mag: 59646, sign: false },
            FP16x16 { mag: 55690, sign: false },
            FP16x16 { mag: 181170, sign: true },
            FP16x16 { mag: 22719, sign: true },
            FP16x16 { mag: 91746, sign: false },
            FP16x16 { mag: 141435, sign: true },
            FP16x16 { mag: 123843, sign: true },
        ]
            .span()
    }
}
