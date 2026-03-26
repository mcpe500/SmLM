#!/usr/bin/env python3
"""
Generate a minimal test model for C++ engine testing.
No PyTorch required - creates random weights directly.
"""

import struct
import argparse
import math

MAGIC = b'SmLM'
VERSION = 1


def create_test_model(output_path: str, 
                      vocab_size: int = 1000,
                      hidden_size: int = 128,
                      num_layers: int = 2,
                      num_heads: int = 4,
                      intermediate_size: int = 512,
                      max_positions: int = 256):
    """
    Create a minimal test model with random weights.
    This is for testing the C++ engine without needing PyTorch.
    """
    
    print(f"Creating test model:")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Heads: {num_heads}")
    print(f"  - Intermediate size: {intermediate_size}")
    print(f"  - Max positions: {max_positions}")
    
    # Calculate params
    embed_params = vocab_size * hidden_size
    pos_params = max_positions * hidden_size
    
    per_layer_params = (
        # QKV projections
        hidden_size * hidden_size * 3 + hidden_size * 3 +
        # Attention output
        hidden_size * hidden_size + hidden_size +
        # Layer norms
        hidden_size * 2 * 2 +
        # MLP
        hidden_size * intermediate_size * 2 + intermediate_size + hidden_size
    )
    
    layer_params = per_layer_params * num_layers
    
    # Final norm + LM head
    final_params = hidden_size * 2 + hidden_size * vocab_size + vocab_size
    
    total_params = embed_params + pos_params + layer_params + final_params
    
    print(f"\nEstimated parameters: {total_params:,}")
    print(f"Model size (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")
    
    import random
    random.seed(42)  # Reproducible
    
    def random_tensor(size: int) -> list:
        """Generate random float32 values with Xavier initialization."""
        scale = math.sqrt(2.0 / size)
        return [random.gauss(0, scale) for _ in range(size)]
    
    def write_tensor(f, data: list):
        """Write tensor to file (little-endian)."""
        f.write(struct.pack('<I', len(data)))
        if data:
            f.write(struct.pack(f'<{len(data)}f', *data))
    
    with open(output_path, 'wb') as f:
        # Magic - write as little-endian uint32
        # "SmLM" as uint32 little-endian = 0x4D4C6D53
        f.write(struct.pack('<I', 0x4D4C6D53))
        f.write(struct.pack('<I', VERSION))
        
        # Config (all little-endian uint32)
        f.write(struct.pack('<I', vocab_size))
        f.write(struct.pack('<I', max_positions))
        f.write(struct.pack('<I', hidden_size))
        f.write(struct.pack('<I', num_layers))
        f.write(struct.pack('<I', num_heads))
        f.write(struct.pack('<I', intermediate_size))
        
        # Token embeddings
        write_tensor(f, random_tensor(embed_params))
        
        # Position embeddings
        write_tensor(f, random_tensor(pos_params))
        
        # Layers
        for layer_idx in range(num_layers):
            layer = {}
            
            # QKV
            layer['q_weight'] = random_tensor(hidden_size * hidden_size)
            layer['q_bias'] = [0.0] * hidden_size
            layer['k_weight'] = random_tensor(hidden_size * hidden_size)
            layer['k_bias'] = [0.0] * hidden_size
            layer['v_weight'] = random_tensor(hidden_size * hidden_size)
            layer['v_bias'] = [0.0] * hidden_size
            
            # Attention output
            layer['attn_out_weight'] = random_tensor(hidden_size * hidden_size)
            layer['attn_out_bias'] = [0.0] * hidden_size
            
            # Layer norms (initialize to 1 and 0)
            layer['ln1_weight'] = [1.0] * hidden_size
            layer['ln1_bias'] = [0.0] * hidden_size
            layer['ln2_weight'] = [1.0] * hidden_size
            layer['ln2_bias'] = [0.0] * hidden_size
            
            # MLP
            layer['mlp_up_weight'] = random_tensor(hidden_size * intermediate_size)
            layer['mlp_up_bias'] = [0.0] * intermediate_size
            layer['mlp_down_weight'] = random_tensor(intermediate_size * hidden_size)
            layer['mlp_down_bias'] = [0.0] * hidden_size
            
            # Write in expected order
            write_tensor(f, layer['q_weight'])
            write_tensor(f, layer['q_bias'])
            write_tensor(f, layer['k_weight'])
            write_tensor(f, layer['k_bias'])
            write_tensor(f, layer['v_weight'])
            write_tensor(f, layer['v_bias'])
            write_tensor(f, layer['attn_out_weight'])
            write_tensor(f, layer['attn_out_bias'])
            write_tensor(f, layer['ln1_weight'])
            write_tensor(f, layer['ln1_bias'])
            write_tensor(f, layer['mlp_up_weight'])
            write_tensor(f, layer['mlp_up_bias'])
            write_tensor(f, layer['mlp_down_weight'])
            write_tensor(f, layer['mlp_down_bias'])
            write_tensor(f, layer['ln2_weight'])
            write_tensor(f, layer['ln2_bias'])
        
        # Final norm
        write_tensor(f, [1.0] * hidden_size)  # weight
        write_tensor(f, [0.0] * hidden_size)  # bias
        
        # LM head
        write_tensor(f, random_tensor(hidden_size * vocab_size))
        write_tensor(f, [0.0] * vocab_size)
    
    # Report
    import os
    file_size = os.path.getsize(output_path)
    print(f"\nModel saved to: {output_path}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"\nTest with:")
    print(f"  ./engine --model {output_path} --input 0,1,2 --max_tokens 10")


def main():
    parser = argparse.ArgumentParser(description='Generate test model for C++ engine')
    parser.add_argument('--output', type=str, default='test_model.slm',
                       help='Output path')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden size')
    parser.add_argument('--layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--intermediate-size', type=int, default=512,
                       help='MLP intermediate size')
    parser.add_argument('--max-positions', type=int, default=256,
                       help='Max position embeddings')
    
    args = parser.parse_args()
    
    create_test_model(
        args.output,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_heads=args.heads,
        intermediate_size=args.intermediate_size,
        max_positions=args.max_positions
    )


if __name__ == '__main__':
    main()
