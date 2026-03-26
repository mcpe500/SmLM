#!/usr/bin/env python3
"""
Export PyTorch transformer weights to pure C++ binary format (.slm)

This script extracts weights from a HuggingFace model and saves them
in a simple binary format that can be loaded by the C++ inference engine.

Usage:
    python export_to_cpp.py --model HuggingFaceTB/SmolLM2-135M --output model.slm
    python export_to_cpp.py --checkpoint checkpoints/model.pt --output model.slm
"""

import argparse
import struct
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Will only be able to convert from .pt checkpoints.")


# Magic number and version for .slm format
MAGIC = b'SmLM'
VERSION = 1


def save_cpp_model(output_path: str, weights: dict, config: dict):
    """
    Save model weights in C++-readable binary format.
    
    File format:
    - magic: 4 bytes = "SmLM"
    - version: 4 bytes = 1
    - config: 6 x int32 (vocab_size, max_positions, hidden_size, num_layers, num_heads, intermediate_size)
    - weights: sequence of (size: uint32, data: float32[size])
    """
    
    with open(output_path, 'wb') as f:
        # Write magic (little-endian uint32: "SmLM" = 0x4D4C6D53)
        f.write(struct.pack('<I', 0x4D4C6D53))
        
        # Write version (little-endian uint32)
        f.write(struct.pack('<I', VERSION))
        
        # Write config (all little-endian uint32)
        f.write(struct.pack('<I', config['vocab_size']))
        f.write(struct.pack('<I', config['max_position_embeddings']))
        f.write(struct.pack('<I', config['hidden_size']))
        f.write(struct.pack('<I', config['num_hidden_layers']))
        f.write(struct.pack('<I', config['num_attention_heads']))
        f.write(struct.pack('<I', config['intermediate_size']))
        
        # Helper to write a tensor as flat float array (little-endian)
        def write_tensor(tensor, name=None):
            if tensor is None:
                # Write empty tensor
                f.write(struct.pack('<I', 0))
                return
            
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().float().numpy()
            else:
                data = tensor.flatten().astype('<f4')  # little-endian float32
            f.write(struct.pack('<I', len(data)))
            f.write(data.tobytes())
        
        # Write weights in the exact order expected by C++ engine
        
        # Token embeddings
        write_tensor(weights.get('token_embeddings'))
        
        # Position embeddings
        write_tensor(weights.get('position_embeddings'))
        
        # Per-layer weights
        layers = weights.get('layers', [])
        for layer in layers:
            write_tensor(layer.get('attention_query_weight'))
            write_tensor(layer.get('attention_query_bias'))
            write_tensor(layer.get('attention_key_weight'))
            write_tensor(layer.get('attention_key_bias'))
            write_tensor(layer.get('attention_value_weight'))
            write_tensor(layer.get('attention_value_bias'))
            write_tensor(layer.get('attention_output_weight'))
            write_tensor(layer.get('attention_output_bias'))
            write_tensor(layer.get('ln1_weight'))
            write_tensor(layer.get('ln1_bias'))
            write_tensor(layer.get('mlp_up_weight'))
            write_tensor(layer.get('mlp_up_bias'))
            write_tensor(layer.get('mlp_down_weight'))
            write_tensor(layer.get('mlp_down_bias'))
            write_tensor(layer.get('ln2_weight'))
            write_tensor(layer.get('ln2_bias'))
        
        # Final layer norm
        write_tensor(weights.get('final_norm_weight'))
        write_tensor(weights.get('final_norm_bias'))
        
        # LM head
        write_tensor(weights.get('lm_head_weight'))
        write_tensor(weights.get('lm_head_bias'))
    
    # Calculate file size
    file_size = Path(output_path).stat().st_size
    print(f"Model saved to: {output_path}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    
    # Count parameters
    total_params = config.get('total_params', 0)
    if total_params > 0:
        print(f"Total parameters: {total_params:,}")
        print(f"Model size (FP32): {total_params * 4 / 1024 / 1024:.2f} MB")


def extract_from_hf(model_name: str, output_path: str):
    """Extract weights from HuggingFace model."""
    
    if not HAS_TORCH:
        print("Error: PyTorch required to load HuggingFace models")
        sys.exit(1)
    
    print(f"Loading model: {model_name}")
    
    # Load model in CPU mode
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True
    )
    
    config = model.config
    
    print(f"Model config:")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Max positions: {config.max_position_embeddings}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Heads: {config.num_attention_heads}")
    print(f"  - Intermediate size: {config.intermediate_size}")
    
    # Extract weights based on model architecture
    # This handles GPT-2 style models (like SmolLM2)
    
    model_dict = model.state_dict()
    
    weights = {}
    
    # Token embeddings
    if 'transformer.wte.weight' in model_dict:
        weights['token_embeddings'] = model_dict['transformer.wte.weight']
    elif 'embed_tokens.weight' in model_dict:
        weights['token_embeddings'] = model_dict['embed_tokens.weight']
    
    # Position embeddings
    if 'transformer.wpe.weight' in model_dict:
        weights['position_embeddings'] = model_dict['transformer.wpe.weight']
    elif 'embed_positions.weight' in model_dict:
        weights['position_embeddings'] = model_dict['embed_positions.weight']
    
    # Transformer layers
    layers = []
    
    # Try different layer naming conventions
    layer_prefixes = [
        'transformer.h.',      # GPT-2 / SmolLM2
        'model.layers.',       # Llama
        'layers.',             # Generic
    ]
    
    # Find which prefix works
    active_prefix = None
    for prefix in layer_prefixes:
        test_key = f"{prefix}0.attn.weight"
        if test_key in model_dict or f"{prefix}0.attention.q_proj.weight" in model_dict:
            active_prefix = prefix.rstrip('.')
            break
    
    if active_prefix is None:
        # Try to find any layer keys
        for key in model_dict.keys():
            if '.layers.' in key or '.h.' in key:
                # Extract prefix
                parts = key.split('.')
                for i, p in enumerate(parts):
                    if p.isdigit():
                        active_prefix = '.'.join(parts[:i])
                        break
                if active_prefix:
                    break
    
    if active_prefix is None:
        print("Warning: Could not detect layer structure. Using generic extraction.")
        # Fallback: try to extract what we can
        for key, value in model_dict.items():
            weights[key] = value
    else:
        print(f"Detected layer prefix: {active_prefix}")
        
        num_layers = config.num_hidden_layers
        
        for layer_idx in range(num_layers):
            layer = {}
            
            # Attention weights - adapt to model architecture
            # For GPT-2 / SmolLM2 style
            if f'{active_prefix}.{layer_idx}.attn.c_attn.weight' in model_dict:
                # Combined QKV projection (GPT-2 style)
                qkv_weight = model_dict[f'{active_prefix}.{layer_idx}.attn.c_attn.weight']
                qkv_bias = model_dict.get(f'{active_prefix}.{layer_idx}.attn.c_attn.bias')
                
                hidden_size = config.hidden_size
                # Split QKV
                q_weight = qkv_weight[:, :hidden_size]
                k_weight = qkv_weight[:, hidden_size:hidden_size*2]
                v_weight = qkv_weight[:, hidden_size*2:hidden_size*3]
                
                if qkv_bias is not None:
                    q_bias = qkv_bias[:hidden_size]
                    k_bias = qkv_bias[hidden_size:hidden_size*2]
                    v_bias = qkv_bias[hidden_size*2:hidden_size*3]
                else:
                    q_bias = k_bias = v_bias = None
                
                layer['attention_query_weight'] = q_weight.t()
                layer['attention_key_weight'] = k_weight.t()
                layer['attention_value_weight'] = v_weight.t()
                layer['attention_query_bias'] = q_bias
                layer['attention_key_bias'] = k_bias
                layer['attention_value_bias'] = v_bias
                
                # Output projection
                layer['attention_output_weight'] = model_dict[f'{active_prefix}.{layer_idx}.attn.c_proj.weight'].t()
                layer['attention_output_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.attn.c_proj.bias')
                
            # For transformer style with separate Q, K, V
            elif f'{active_prefix}.{layer_idx}.self_attn.q_proj.weight' in model_dict:
                layer['attention_query_weight'] = model_dict[f'{active_prefix}.{layer_idx}.self_attn.q_proj.weight'].t()
                layer['attention_key_weight'] = model_dict[f'{active_prefix}.{layer_idx}.self_attn.k_proj.weight'].t()
                layer['attention_value_weight'] = model_dict[f'{active_prefix}.{layer_idx}.self_attn.v_proj.weight'].t()
                layer['attention_query_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.self_attn.q_proj.bias')
                layer['attention_key_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.self_attn.k_proj.bias')
                layer['attention_value_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.self_attn.v_proj.bias')
                layer['attention_output_weight'] = model_dict[f'{active_prefix}.{layer_idx}.self_attn.o_proj.weight'].t()
                layer['attention_output_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.self_attn.o_proj.bias')
            
            # Layer norms
            if f'{active_prefix}.{layer_idx}.ln_1.weight' in model_dict:
                # GPT-2 style
                layer['ln1_weight'] = model_dict[f'{active_prefix}.{layer_idx}.ln_1.weight']
                layer['ln1_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.ln_1.bias')
                layer['ln2_weight'] = model_dict[f'{active_prefix}.{layer_idx}.ln_2.weight']
                layer['ln2_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.ln_2.bias')
            elif f'{active_prefix}.{layer_idx}.input_layernorm.weight' in model_dict:
                # Llama style
                layer['ln1_weight'] = model_dict[f'{active_prefix}.{layer_idx}.input_layernorm.weight']
                layer['ln1_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.input_layernorm.bias')
                layer['ln2_weight'] = model_dict[f'{active_prefix}.{layer_idx}.post_attention_layernorm.weight']
                layer['ln2_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.post_attention_layernorm.bias')
            
            # MLP
            if f'{active_prefix}.{layer_idx}.mlp.c_fc.weight' in model_dict:
                # GPT-2 style
                layer['mlp_up_weight'] = model_dict[f'{active_prefix}.{layer_idx}.mlp.c_fc.weight'].t()
                layer['mlp_up_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.mlp.c_fc.bias')
                layer['mlp_down_weight'] = model_dict[f'{active_prefix}.{layer_idx}.mlp.c_proj.weight'].t()
                layer['mlp_down_bias'] = model_dict.get(f'{active_prefix}.{layer_idx}.mlp.c_proj.bias')
            elif f'{active_prefix}.{layer_idx}.mlp.gate_proj.weight' in model_dict:
                # Llama style (with gating)
                layer['mlp_gate_weight'] = model_dict[f'{active_prefix}.{layer_idx}.mlp.gate_proj.weight'].t()
                layer['mlp_up_weight'] = model_dict[f'{active_prefix}.{layer_idx}.mlp.up_proj.weight'].t()
                layer['mlp_down_weight'] = model_dict[f'{active_prefix}.{layer_idx}.mlp.down_proj.weight'].t()
                layer['mlp_up_bias'] = None
                layer['mlp_down_bias'] = None
            
            layers.append(layer)
        
        weights['layers'] = layers
    
    # Final layer norm
    if 'transformer.ln_f.weight' in model_dict:
        weights['final_norm_weight'] = model_dict['transformer.ln_f.weight']
        weights['final_norm_bias'] = model_dict.get('transformer.ln_f.bias')
    elif 'model.norm.weight' in model_dict:
        weights['final_norm_weight'] = model_dict['model.norm.weight']
        weights['final_norm_bias'] = model_dict.get('model.norm.bias')
    
    # LM head
    if 'lm_head.weight' in model_dict:
        weights['lm_head_weight'] = model_dict['lm_head.weight'].t()
        weights['lm_head_bias'] = model_dict.get('lm_head.bias')
    elif 'embed_tokens.weight' in model_dict and config.tie_word_embeddings:
        # Tied embeddings - use token embeddings as LM head
        print("Using tied embeddings for LM head")
        weights['lm_head_weight'] = weights['token_embeddings'].clone()
        weights['lm_head_bias'] = None
    else:
        # Create zero LM head as placeholder
        print("Warning: LM head not found, creating placeholder")
        weights['lm_head_weight'] = torch.zeros(config.hidden_size, config.vocab_size)
        weights['lm_head_bias'] = torch.zeros(config.vocab_size)
    
    # Build config dict
    config_dict = {
        'vocab_size': config.vocab_size,
        'max_position_embeddings': config.max_position_embeddings,
        'hidden_size': config.hidden_size,
        'num_hidden_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads,
        'intermediate_size': config.intermediate_size,
    }
    
    # Count total params
    total_params = 0
    for name, tensor in weights.items():
        if isinstance(tensor, torch.Tensor):
            total_params += tensor.numel()
        elif isinstance(tensor, list):
            for t in tensor:
                if isinstance(t, torch.Tensor):
                    total_params += t.numel()
    config_dict['total_params'] = total_params
    
    # Save
    save_cpp_model(output_path, weights, config_dict)
    print("\nExport complete! Use with:")
    print(f"  ./engine --model {output_path}")


def extract_from_checkpoint(checkpoint_path: str, output_path: str, config_path: str = None):
    """Extract weights from PyTorch checkpoint."""
    
    if not HAS_TORCH:
        print("Error: PyTorch required to load checkpoints")
        sys.exit(1)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load config
    if config_path and Path(config_path).exists():
        import json
        with open(config_path) as f:
            config_dict = json.load(f)
    elif 'config' in checkpoint:
        config_dict = checkpoint['config']
    else:
        # Try to infer from weights
        print("Warning: No config found, inferring from weights")
        config_dict = infer_config_from_weights(state_dict)
    
    # Save
    save_cpp_model(output_path, state_dict, config_dict)


def infer_config_from_weights(state_dict):
    """Try to infer model config from weight shapes."""
    config = {
        'vocab_size': 50257,  # Default GPT-2
        'max_position_embeddings': 1024,
        'hidden_size': 512,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 2048,
    }
    
    # Try to find embedding dim
    for name, tensor in state_dict.items():
        if 'embed' in name.lower() and len(tensor.shape) == 2:
            config['vocab_size'] = tensor.shape[0]
            config['hidden_size'] = tensor.shape[1]
            break
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to C++ binary format')
    parser.add_argument('--model', type=str, help='HuggingFace model name or path')
    parser.add_argument('--checkpoint', type=str, help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--config', type=str, help='Path to config JSON (for checkpoint mode)')
    parser.add_argument('--output', type=str, required=True, help='Output path for .slm file')
    parser.add_argument('--student-config', type=str, help='Student config YAML (optional)')
    
    args = parser.parse_args()
    
    if not args.model and not args.checkpoint:
        print("Error: Must specify either --model or --checkpoint")
        parser.print_help()
        sys.exit(1)
    
    if args.model:
        extract_from_hf(args.model, args.output)
    elif args.checkpoint:
        extract_from_checkpoint(args.checkpoint, args.output, args.config)


if __name__ == '__main__':
    main()
