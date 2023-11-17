#[starknet::contract]
mod OrionRunner {
    use core::array::SpanTrait;
    use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor, FP16x16TensorAdd};
    use orion::operators::nn::{NNTrait, FP16x16NN};
    use orion::numbers::{FP16x16, FixedTrait};

    use sequential_1_dense_1_matmul_readvariableop_0::tensor as t1;
    use sequential_1_dense_1_biasadd_readvariableop_0::tensor as t2;
    use sequential_1_dense_2_matmul_readvariableop_0::tensor as t3;
    use sequential_1_dense_2_biasadd_readvariableop_0::tensor as t4;
    use sequential_1_dense_3_matmul_readvariableop_0::tensor as t5;
    use sequential_1_dense_3_biasadd_readvariableop_0::tensor as t6;

    #[storage]
    struct Storage {
        id: u8,
    }

    #[external(v0)]
    fn predict(self: @ContractState, mut x: Tensor<FP16x16>) -> FP16x16 {
        // let two = FixedTrait::<FP16x16>::new_unscaled(2, false);
        // let mut x = Tensor {
        //     shape: array![9].span(),
        //     data: array![two, two, two, two, two, two, two, two, two].span()
        // };

        // DENSE 1
        x = TensorTrait::matmul(@x, @t1());
        x = x + t2();
        x = NNTrait::relu(@x);

        // DENSE 2
        x = TensorTrait::matmul(@x, @t3());
        x = x + t4();
        x = NNTrait::relu(@x);

        // DENSE 3
        x = TensorTrait::matmul(@x, @t5());
        x = x + t6();

        return *x.data.at(0);
    }
}
