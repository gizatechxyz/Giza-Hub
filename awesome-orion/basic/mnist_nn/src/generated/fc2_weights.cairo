use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::numbers::i32;


fn fc2_weights() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);
    shape.append(10);
    let mut data = ArrayTrait::<i32>::new();
    data.append(i32 { mag: 42, sign: true });
    data.append(i32 { mag: 41, sign: false });
    data.append(i32 { mag: 15, sign: false });
    data.append(i32 { mag: 27, sign: true });
    data.append(i32 { mag: 58, sign: true });
    data.append(i32 { mag: 71, sign: false });
    data.append(i32 { mag: 21, sign: true });
    data.append(i32 { mag: 50, sign: true });
    data.append(i32 { mag: 32, sign: false });
    data.append(i32 { mag: 21, sign: true });
    data.append(i32 { mag: 38, sign: true });
    data.append(i32 { mag: 67, sign: true });
    data.append(i32 { mag: 35, sign: true });
    data.append(i32 { mag: 112, sign: true });
    data.append(i32 { mag: 95, sign: false });
    data.append(i32 { mag: 78, sign: false });
    data.append(i32 { mag: 15, sign: false });
    data.append(i32 { mag: 28, sign: true });
    data.append(i32 { mag: 64, sign: false });
    data.append(i32 { mag: 49, sign: false });
    data.append(i32 { mag: 19, sign: false });
    data.append(i32 { mag: 69, sign: true });
    data.append(i32 { mag: 53, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 62, sign: true });
    data.append(i32 { mag: 47, sign: false });
    data.append(i32 { mag: 30, sign: true });
    data.append(i32 { mag: 70, sign: true });
    data.append(i32 { mag: 28, sign: true });
    data.append(i32 { mag: 48, sign: false });
    data.append(i32 { mag: 69, sign: true });
    data.append(i32 { mag: 21, sign: true });
    data.append(i32 { mag: 35, sign: false });
    data.append(i32 { mag: 38, sign: true });
    data.append(i32 { mag: 100, sign: true });
    data.append(i32 { mag: 41, sign: true });
    data.append(i32 { mag: 13, sign: true });
    data.append(i32 { mag: 78, sign: false });
    data.append(i32 { mag: 12, sign: true });
    data.append(i32 { mag: 29, sign: false });
    data.append(i32 { mag: 59, sign: false });
    data.append(i32 { mag: 49, sign: true });
    data.append(i32 { mag: 36, sign: true });
    data.append(i32 { mag: 12, sign: false });
    data.append(i32 { mag: 11, sign: false });
    data.append(i32 { mag: 24, sign: false });
    data.append(i32 { mag: 14, sign: false });
    data.append(i32 { mag: 31, sign: false });
    data.append(i32 { mag: 19, sign: true });
    data.append(i32 { mag: 99, sign: true });
    data.append(i32 { mag: 6, sign: false });
    data.append(i32 { mag: 11, sign: false });
    data.append(i32 { mag: 29, sign: false });
    data.append(i32 { mag: 9, sign: false });
    data.append(i32 { mag: 2, sign: true });
    data.append(i32 { mag: 127, sign: true });
    data.append(i32 { mag: 117, sign: true });
    data.append(i32 { mag: 31, sign: false });
    data.append(i32 { mag: 39, sign: false });
    data.append(i32 { mag: 17, sign: true });
    data.append(i32 { mag: 67, sign: false });
    data.append(i32 { mag: 9, sign: false });
    data.append(i32 { mag: 42, sign: false });
    data.append(i32 { mag: 112, sign: true });
    data.append(i32 { mag: 26, sign: false });
    data.append(i32 { mag: 10, sign: true });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 73, sign: true });
    data.append(i32 { mag: 21, sign: false });
    data.append(i32 { mag: 65, sign: true });
    data.append(i32 { mag: 76, sign: true });
    data.append(i32 { mag: 5, sign: true });
    data.append(i32 { mag: 90, sign: true });
    data.append(i32 { mag: 19, sign: false });
    data.append(i32 { mag: 75, sign: true });
    data.append(i32 { mag: 36, sign: true });
    data.append(i32 { mag: 71, sign: false });
    data.append(i32 { mag: 45, sign: true });
    data.append(i32 { mag: 82, sign: false });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 5, sign: false });
    data.append(i32 { mag: 81, sign: false });
    data.append(i32 { mag: 12, sign: false });
    data.append(i32 { mag: 13, sign: true });
    data.append(i32 { mag: 22, sign: false });
    data.append(i32 { mag: 28, sign: true });
    data.append(i32 { mag: 46, sign: true });
    data.append(i32 { mag: 1, sign: false });
    data.append(i32 { mag: 110, sign: true });
    data.append(i32 { mag: 3, sign: true });
    data.append(i32 { mag: 82, sign: true });
    data.append(i32 { mag: 16, sign: false });
    data.append(i32 { mag: 32, sign: false });
    data.append(i32 { mag: 12, sign: false });
    data.append(i32 { mag: 31, sign: false });
    data.append(i32 { mag: 2, sign: false });
    data.append(i32 { mag: 45, sign: false });
    data.append(i32 { mag: 30, sign: true });
    data.append(i32 { mag: 87, sign: true });
    data.append(i32 { mag: 125, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
