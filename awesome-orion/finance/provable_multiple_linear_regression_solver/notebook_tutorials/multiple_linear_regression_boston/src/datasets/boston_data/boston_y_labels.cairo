use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq };
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn boston_y_labels() ->  Tensor<FP16x16>  {
    let tensor = TensorTrait::<FP16x16>::new( 
    shape: array![50].span(),
    data: array![ 
    FixedTrait::new(1422131, false ),
    FixedTrait::new(1199308, false ),
    FixedTrait::new(1638400, false ),
    FixedTrait::new(878182, false ),
    FixedTrait::new(1251737, false ),
    FixedTrait::new(1120665, false ),
    FixedTrait::new(2175795, false ),
    FixedTrait::new(688128, false ),
    FixedTrait::new(1284505, false ),
    FixedTrait::new(1651507, false ),
    FixedTrait::new(1284505, false ),
    FixedTrait::new(3276800, false ),
    FixedTrait::new(904396, false ),
    FixedTrait::new(1625292, false ),
    FixedTrait::new(1638400, false ),
    FixedTrait::new(1448345, false ),
    FixedTrait::new(2044723, false ),
    FixedTrait::new(1330380, false ),
    FixedTrait::new(1559756, false ),
    FixedTrait::new(976486, false ),
    FixedTrait::new(1323827, false ),
    FixedTrait::new(1546649, false ),
    FixedTrait::new(1310720, false ),
    FixedTrait::new(3198156, false ),
    FixedTrait::new(668467, false ),
    FixedTrait::new(1415577, false ),
    FixedTrait::new(1526988, false ),
    FixedTrait::new(2437939, false ),
    FixedTrait::new(2031616, false ),
    FixedTrait::new(1435238, false ),
    FixedTrait::new(1166540, false ),
    FixedTrait::new(1841561, false ),
    FixedTrait::new(1481113, false ),
    FixedTrait::new(3014656, false ),
    FixedTrait::new(1022361, false ),
    FixedTrait::new(1225523, false ),
    FixedTrait::new(1428684, false ),
    FixedTrait::new(1743257, false ),
    FixedTrait::new(1238630, false ),
    FixedTrait::new(2188902, false ),
    FixedTrait::new(1618739, false ),
    FixedTrait::new(1985740, false ),
    FixedTrait::new(1369702, false ),
    FixedTrait::new(1520435, false ),
    FixedTrait::new(1625292, false ),
    FixedTrait::new(865075, false ),
    FixedTrait::new(1717043, false ),
    FixedTrait::new(1533542, false ),
    FixedTrait::new(635699, false ),
    FixedTrait::new(1828454, false ),
].span() 
 
);

return tensor; 
}