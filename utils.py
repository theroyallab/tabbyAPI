def add_args(parser):
    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory")
    parser.add_argument("-gs", "--gpu_split", type = str, help = "\"auto\", or VRAM allocation per GPU in GB")
    parser.add_argument("-l", "--max_seq_len", type = int, help = "Maximum sequence length")
    parser.add_argument("-rs", "--rope_scale", type = float, default = 1.0, help = "RoPE scaling factor")
    parser.add_argument("-ra", "--rope_alpha", type = float, default = 1.0, help = "RoPE alpha value (NTK)")
    parser.add_argument("-nfa", "--no_flash_attn", action = "store_true", help = "Disable Flash Attention")
    parser.add_argument("-lm", "--low_mem", action = "store_true", help = "Enable VRAM optimizations, potentially trading off speed")
