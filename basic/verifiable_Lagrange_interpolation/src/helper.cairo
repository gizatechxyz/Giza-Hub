use traits::TryInto;
use alexandria_data_structures::array_ext::{SpanTraitExt};
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{Tensor, TensorTrait};
use orion::numbers::fixed_point::{core::{FixedTrait}};

use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorDiv};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16, FP16x16Impl, FP16x16Add, FP16x16AddEq, FP16x16Sub, FP16x16Mul, FP16x16MulEq,
    FP16x16TryIntoU128, FP16x16PartialEq, FP16x16PartialOrd, FP16x16SubEq, FP16x16Neg, FP16x16Div,
    FP16x16IntoFelt252, FP16x16Print, HALF
};

use orion::numbers::fixed_point::implementations::fp16x16::math::trig;

fn lagrange_interpolation(x_interpolated: @Tensor<FP16x16>, X: @Tensor<FP16x16>, Y: @Tensor<FP16x16>) -> Tensor<FP16x16> {

    let n = ((*X).data.len());
    let m = ((*x_interpolated).data.len());

    let mut y_data = ArrayTrait::<FP16x16>::new();
    let mut phi = ArrayTrait::<FP16x16>::new(); 
    let mut j = 0;
    
    loop {
        if j == m {
            break;
        }
        let mut y_j = FixedTrait::new(0,false);
        let mut i = 0;
        loop {
            if i == n {
                break;
            }
            let mut phi_i = FixedTrait::<FP16x16>::new(65536,false);
            let mut k = 0;
            loop {
                if k == n {
                    break;
                }
                if i != k {
                    phi_i = phi_i * (*(*x_interpolated).data.at(j) - *(*X).data.at(k))/( *(*X).data.at(i) -  *(*X).data.at(k))
                }
                k += 1;
            };
            y_j = y_j +  *(*Y).data.at(i) * phi_i;
            i += 1;
        };
        y_data.append(y_j);
        j += 1;

    };

    return TensorTrait::new((*x_interpolated).shape, y_data.span());

}
