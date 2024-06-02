#[cfg(test)]
mod tests {
    use traits::TryInto;
    use alexandria_data_structures::array_ext::{SpanTraitExt};
    use array::{ArrayTrait, SpanTrait};
    use orion::operators::tensor::{Tensor, TensorTrait};
    use orion::numbers::fixed_point::{core::{FixedTrait}};

    use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorDiv, FP16x16TensorSub};

    use orion::numbers::fixed_point::implementations::fp16x16::core::{
        FP16x16, FP16x16Impl, FP16x16Add, FP16x16AddEq, FP16x16Sub, FP16x16Mul, FP16x16MulEq,
        FP16x16TryIntoU128, FP16x16PartialEq, FP16x16PartialOrd, FP16x16SubEq, FP16x16Neg,
        FP16x16Div, FP16x16IntoFelt252, FP16x16Print
    };

    use lagrange::helper::lagrange_interpolation;
    use lagrange::generated::{X::X, Y::Y, x::x, y::y};

    #[test]
    #[available_gas(99999999999999999)]
    fn lagrange_test() {
        let tol = FixedTrait::<FP16x16>::new(655, false); // 655 is 0.01 = 1e-2
        let max_iter = 500_usize;

        // Nodes :
        let X = X();
        let Y = Y();

        let x = x();
        let y_expected = y();

        let y_actual = lagrange_interpolation(@x, @X, @Y);

        let mut i = 0;
        loop {
            if i == y_expected.data.len() {
                break;
            }

            assert(*y_expected.data.at(i) - *y_actual.data.at(i) < tol, 'difference below threshold');
            i += 1;
        }
    }
}
