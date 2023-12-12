use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq };
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn x_feature_data() ->  Tensor<FP16x16>  {
    let tensor = TensorTrait::<FP16x16>::new( 
    shape: array![50].span(),
    data: array![ 
    FixedTrait::new(90639, false ),
    FixedTrait::new(12581, true ),
    FixedTrait::new(33595, false ),
    FixedTrait::new(92893, false ),
    FixedTrait::new(64841, false ),
    FixedTrait::new(21784, false ),
    FixedTrait::new(93600, false ),
    FixedTrait::new(139107, false ),
    FixedTrait::new(46680, true ),
    FixedTrait::new(148678, true ),
    FixedTrait::new(55700, false ),
    FixedTrait::new(63442, false ),
    FixedTrait::new(16625, false ),
    FixedTrait::new(15088, false ),
    FixedTrait::new(109945, false ),
    FixedTrait::new(22098, false ),
    FixedTrait::new(28923, false ),
    FixedTrait::new(55032, true ),
    FixedTrait::new(29968, false ),
    FixedTrait::new(17353, false ),
    FixedTrait::new(126, true ),
    FixedTrait::new(6705, true ),
    FixedTrait::new(81234, true ),
    FixedTrait::new(38498, true ),
    FixedTrait::new(75536, true ),
    FixedTrait::new(984, true ),
    FixedTrait::new(45491, true ),
    FixedTrait::new(88496, false ),
    FixedTrait::new(8992, false ),
    FixedTrait::new(28549, false ),
    FixedTrait::new(61676, true ),
    FixedTrait::new(54096, true ),
    FixedTrait::new(91046, false ),
    FixedTrait::new(53660, false ),
    FixedTrait::new(6145, true ),
    FixedTrait::new(26994, false ),
    FixedTrait::new(90657, false ),
    FixedTrait::new(21638, true ),
    FixedTrait::new(50848, false ),
    FixedTrait::new(4550, true ),
    FixedTrait::new(7560, true ),
    FixedTrait::new(41550, false ),
    FixedTrait::new(200, false ),
    FixedTrait::new(102341, false ),
    FixedTrait::new(25789, false ),
    FixedTrait::new(9158, false ),
    FixedTrait::new(102276, true ),
    FixedTrait::new(76823, true ),
    FixedTrait::new(69440, true ),
    FixedTrait::new(17547, true ),
].span() 
 
);

return tensor; 
}