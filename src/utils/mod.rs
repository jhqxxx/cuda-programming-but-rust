use cuda_core::LaunchConfig;

pub fn get_launch_config_1d(num_elems: u32, block_size: u32) -> LaunchConfig {
    let grid_x = num_elems.div_ceil(block_size);
    LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}
