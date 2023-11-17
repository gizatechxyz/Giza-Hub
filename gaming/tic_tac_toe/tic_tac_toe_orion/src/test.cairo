#[cfg(test)]
mod tests {
    use core::array::SpanTrait;
    use orion::operators::tensor::{TensorTrait, FP16x16Tensor, Tensor, FP16x16TensorAdd};
    use orion::operators::nn::{NNTrait, FP16x16NN};
    use orion::numbers::{FP16x16, FixedTrait};

    use sequential_1_dense_1_matmul_readvariableop_0::tensor as _sequential_1_dense_1_matmul_readvariableop_0;
    use sequential_1_dense_1_biasadd_readvariableop_0::tensor as _sequential_1_dense_1_biasadd_readvariableop_0;
    use sequential_1_dense_2_matmul_readvariableop_0::tensor as _sequential_1_dense_2_matmul_readvariableop_0;
    use sequential_1_dense_2_biasadd_readvariableop_0::tensor as _sequential_1_dense_2_biasadd_readvariableop_0;
    use sequential_1_dense_3_matmul_readvariableop_0::tensor as _sequential_1_dense_3_matmul_readvariableop_0;
    use sequential_1_dense_3_biasadd_readvariableop_0::tensor as _sequential_1_dense_3_biasadd_readvariableop_0;

    use debug::PrintTrait;

    #[test]
    #[available_gas(2000000000000)]
    fn _main() {
        let two = FixedTrait::<FP16x16>::new_unscaled(2, false);

        let mut x = Tensor {
            shape: array![9].span(),
            data: array![two, two, two, two, two, two, two, two, two].span()
        };

        // DENSE 1
        x = TensorTrait::matmul(@x, @_sequential_1_dense_1_matmul_readvariableop_0());
        x = x + _sequential_1_dense_1_biasadd_readvariableop_0();
        x = NNTrait::relu(@x);

        // DENSE 2
        x = TensorTrait::matmul(@x, @_sequential_1_dense_2_matmul_readvariableop_0());
        x = x + _sequential_1_dense_2_biasadd_readvariableop_0();
        x = NNTrait::relu(@x);

        // DENSE 3
        x = TensorTrait::matmul(@x, @_sequential_1_dense_3_matmul_readvariableop_0());
        x = x + _sequential_1_dense_3_biasadd_readvariableop_0();

        (*x.data.at(0)).print();
    }
}
