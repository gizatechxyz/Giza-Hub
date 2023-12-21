use array::ArrayTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16PartialEq };
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::numbers::{FP16x16, FixedTrait};

fn y_label_data() ->  Tensor<FP16x16>  {
    let tensor = TensorTrait::<FP16x16>::new( 
    shape: array![50].span(),
    data: array![ 
    FixedTrait::new(7282724, false ),
    FixedTrait::new(6435011, false ),
    FixedTrait::new(6662231, false ),
    FixedTrait::new(7271410, false ),
    FixedTrait::new(7099095, false ),
    FixedTrait::new(6751687, false ),
    FixedTrait::new(7403695, false ),
    FixedTrait::new(7831893, false ),
    FixedTrait::new(6135683, false ),
    FixedTrait::new(5448106, false ),
    FixedTrait::new(6992113, false ),
    FixedTrait::new(7129256, false ),
    FixedTrait::new(6678313, false ),
    FixedTrait::new(6524452, false ),
    FixedTrait::new(7538849, false ),
    FixedTrait::new(6685568, false ),
    FixedTrait::new(6749158, false ),
    FixedTrait::new(6149931, false ),
    FixedTrait::new(6876758, false ),
    FixedTrait::new(6623147, false ),
    FixedTrait::new(6679189, false ),
    FixedTrait::new(6578635, false ),
    FixedTrait::new(5894520, false ),
    FixedTrait::new(6161430, false ),
    FixedTrait::new(5887716, false ),
    FixedTrait::new(6440009, false ),
    FixedTrait::new(6209384, false ),
    FixedTrait::new(7208597, false ),
    FixedTrait::new(6679473, false ),
    FixedTrait::new(6809111, false ),
    FixedTrait::new(6068970, false ),
    FixedTrait::new(6089744, false ),
    FixedTrait::new(7360056, false ),
    FixedTrait::new(6971060, false ),
    FixedTrait::new(6419231, false ),
    FixedTrait::new(6780044, false ),
    FixedTrait::new(7279453, false ),
    FixedTrait::new(6350620, false ),
    FixedTrait::new(7023820, false ),
    FixedTrait::new(6568475, false ),
    FixedTrait::new(6528424, false ),
    FixedTrait::new(6936953, false ),
    FixedTrait::new(6511689, false ),
    FixedTrait::new(7367935, false ),
    FixedTrait::new(6860285, false ),
    FixedTrait::new(6800462, false ),
    FixedTrait::new(5650037, false ),
    FixedTrait::new(5915425, false ),
    FixedTrait::new(5913912, false ),
    FixedTrait::new(6491295, false ),
].span() 
 
);

return tensor; 
}