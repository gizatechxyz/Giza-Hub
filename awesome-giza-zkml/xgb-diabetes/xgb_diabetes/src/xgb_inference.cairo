use core::array::ArrayTrait;


#[derive(Copy, Drop)]
pub struct Tree {
    pub base_weights: Span<i32>,
    pub left_children: Span<u32>,
    pub right_children: Span<u32>,
    pub split_indices: Span<u32>,
    pub split_conditions: Span<i32>
}

pub fn navigate_tree_and_accumulate_score(tree: Tree, features: Span<i32>, node: u32) -> i32 {
    if *tree.left_children[node] == 0 {
        if *tree.right_children[node] == 0{
            return *tree.base_weights[node];
        }
    }
    let mut next_node: u32 = 0;
    let feature_index = *tree.split_indices[node];
    let threshold = *tree.split_conditions[node];
    if *features.at(feature_index) < threshold{
        next_node = *tree.left_children[node];
    }
    else{
        next_node = *tree.right_children[node];
    }
    navigate_tree_and_accumulate_score(tree, features, next_node)
}

pub fn accumulate_scores_from_trees(num_trees: u32, trees: Span<Tree>, features: Span<i32>, index:u32, accumulated_score:i32) -> i32{
    if index >= num_trees{
        return accumulated_score;
        }
    let tree: Tree = *trees[index];
    let score_from_tree: i32 = navigate_tree_and_accumulate_score(tree, features, 0);
    accumulate_scores_from_trees(num_trees, trees, features, index + 1, accumulated_score + score_from_tree)
}
