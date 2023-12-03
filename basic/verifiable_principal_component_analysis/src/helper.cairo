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

#[derive(Copy, Drop)]
struct EigenValues<FP16x16> {
    p_index: usize,
    q_index: usize,
    theta: FP16x16,
}

fn div_by_scalar(self: @Tensor<FP16x16>, divisor: u32) -> Tensor<FP16x16> {
    let mut data = (*self).data;
    let mut data_array = ArrayTrait::new();

    loop {
        match data.pop_front() {
            Option::Some(elem) => {
                data_array.append(FixedTrait::new(*elem.mag / divisor, *elem.sign));
            },
            Option::None(_) => {
                break TensorTrait::<FP16x16>::new((*self).shape, data_array.span());
            }
        };
    }
}

fn div_by_fp(self: @Tensor<FP16x16>, divisor: FP16x16) -> Tensor<FP16x16> {
    let mut data = (*self).data;
    let mut data_array = ArrayTrait::new();

    loop {
        match data.pop_front() {
            Option::Some(elem) => { data_array.append(FP16x16Div::div(*elem, divisor)); },
            Option::None(_) => {
                break TensorTrait::<FP16x16>::new((*self).shape, data_array.span());
            }
        };
    }
}

// find_max_off_diag: Finds the maximum off-diagonal element in a square Tensor.
fn find_max_off_diag(a: @Tensor<FP16x16>) -> (usize, usize) {
    let mut data = *a.data;
    let mut shape = *a.shape;

    let n = *(*a).shape.at(0);

    let mut i = 0_usize;
    let mut j = 0_usize;
    let mut p = 0_usize;
    let mut q = 1_usize;

    let mut max_val = FixedTrait::abs((*a).at(indices: array![p, q].span()));

    loop {
        if i == n {
            break (p, q);
        };

        j = i + 1;

        loop {
            if j == n {
                break;
            };
            if FixedTrait::abs((a).at(indices: array![i, j].span())) > max_val {
                max_val = FixedTrait::abs((a).at(indices: array![i, j].span()));
                p = i;
                q = j;
            };
            j += 1;
        };
        i += 1;
    }
}

// jacobi_eigensystem: Implements the Jacobi eigenvalue algorithm to compute the eigenvalues and eigenvectors of a symmetric Tensor.
fn jacobi_eigensystem(
    mut a: Tensor<FP16x16>, tol: FP16x16, max_iter: usize
) -> (Tensor<FP16x16>, Tensor<FP16x16>) {
    assert(
        !((a).shape.len() != 2_usize || ((a).shape.at(0_usize) != (a).shape.at(1_usize))),
        'a must be a square matrix'
    );

    // let two = FixedTrait::new(ONE, false) + FixedTrait::new(ONE, false);
    let two = FixedTrait::ONE() + FixedTrait::ONE();
    let four = two * two;
    let half = FixedTrait::<FP16x16>::new(HALF, false);
    let pi = FixedTrait::<FP16x16>::new(trig::PI, false);

    let mut data = a.data;
    let mut shape = a.shape;
    let numRows = *((shape).at(0));
    let mut v = eye(numRows: numRows);

    let mut i: usize = 0;

    loop {
        let (p, q) = find_max_off_diag(@a);

        if i == max_iter || FixedTrait::abs((a).at(indices: array![p, q].span())) < tol {
            break (extract_diagonal(@a), v);
        };

        let theta = if (a)
            .at(indices: array![p, p].span()) == (a)
            .at(indices: array![q, q].span()) {
            FP16x16Div::div(pi, four)
        } else {
            half
                * trig::atan(
                    FP16x16Div::div(
                        two * (a).at(indices: array![p, q].span()),
                        (FP16x16Sub::sub(
                            (a).at(indices: array![p, p].span()),
                            (a).at(indices: array![q, q].span())
                        ))
                    )
                )
        };

        let eigensystem = EigenValues { p_index: p, q_index: q, theta: theta };

        let j_eye = eye(numRows: numRows);

        let j = update_eigen_values(self: @j_eye, eigensystem: eigensystem);

        let transpose_j = j.transpose(axes: array![1, 0].span());
        a = transpose_j.matmul(@a).matmul(@j);

        v = v.matmul(@j);

        i += 1;
    }
}

// eye: Generates an identity Tensor of the specified size
fn eye(numRows: usize) -> Tensor<FP16x16> {
    let mut data_array = ArrayTrait::new();

    let mut x: usize = 0;

    loop {
        if x == numRows {
            break;
        };

        let mut y: usize = 0;

        loop {
            if y == numRows {
                break;
            };

            if x == y {
                data_array.append(FixedTrait::ONE());
            } else {
                data_array.append(FixedTrait::ZERO());
            };

            y += 1;
        };
        x += 1;
    };

    Tensor::<FP16x16> { shape: array![numRows, numRows].span(), data: data_array.span() }
}

// extract_diagonal: Extracts the diagonal elements from a square tensor
fn extract_diagonal(self: @Tensor<FP16x16>) -> Tensor<FP16x16> {
    let mut data = (*self).data;
    let mut data_array = ArrayTrait::new();

    let dims = (*self).shape.at(0);

    let mut x: usize = 0;

    loop {
        if x == *dims {
            break;
        };

        let mut y: usize = 0;

        loop {
            if y == *dims {
                break;
            };

            match data.pop_front() {
                Option::Some(elem) => { if x == y {
                    data_array.append(*elem);
                }; },
                Option::None(_) => { break; }
            };
            y += 1;
        };
        x += 1;
    };

    Tensor::<FP16x16> { shape: array![*dims].span(), data: data_array.span() }
}

// update_eigen_values: Updates the eigenvalues of a symmetric tensor
fn update_eigen_values(
    self: @Tensor<FP16x16>, eigensystem: EigenValues<FP16x16>
) -> Tensor<FP16x16> {
    let mut data = (*self).data;
    let mut data_array = ArrayTrait::new();

    let mut x: usize = 0;
    let mut y: usize = 0;
    let mut index: usize = 0;
    let dims = (*self).shape.at(0);
    let items = *dims * *dims;
    let dims_y = (*self).shape.at(1);

    loop {
        if index == items {
            break;
        };

        if y == *dims_y {
            x += 1;
            y = 0;
        };

        match data.pop_front() {
            Option::Some(elem) => {
                let eigen_values = eigensystem;

                let value = if (eigen_values.p_index, eigen_values.p_index) == (x, y) {
                    trig::cos(eigen_values.theta)
                } else if (eigen_values.q_index, eigen_values.q_index) == (x, y) {
                    trig::cos(eigen_values.theta)
                } else if (eigen_values.p_index, eigen_values.q_index) == (x, y) {
                    trig::sin(eigen_values.theta)
                } else if (eigen_values.q_index, eigen_values.p_index) == (x, y) {
                    trig::sin(-eigen_values.theta)
                } else {
                    *elem
                };

                data_array.append(value);
                y += 1;
                index += 1;
            },
            Option::None(_) => { break; }
        };
    };

    Tensor::<FP16x16> { shape: *self.shape, data: data_array.span() }
}

// check_unit_diagonal_tensor: Checks whether a square tensor has a unit diagonal
fn check_unit_diagonal_tensor(self: @Tensor<FP16x16>) -> bool {
    let mut x: usize = 0;
    let mut valid: bool = true;
    let dim_x = (*self).shape.at(0);
    let dim_y = (*self).shape.at(1);

    loop {
        if x == *dim_x || !valid {
            break valid;
        };

        let mut y: usize = 0;

        loop {
            if y == *dim_y {
                break;
            };

            if x == y {
                if (self).at(indices: array![x, y].span()) != FixedTrait::ONE() {
                    valid = false;
                    break;
                }
            } else {
                if (self).at(indices: array![x, y].span()) != FixedTrait::ZERO() {
                    valid = false;
                    break;
                }
            };

            y += 1;
        };
        x += 1;
    }
}
