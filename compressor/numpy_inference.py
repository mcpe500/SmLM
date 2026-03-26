"""NumPy-based transformer inference for CPU-only environments."""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax function."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LayerNorm:
    """Layer normalization."""
    
    def __init__(self, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
        self.weight = weight
        self.bias = bias
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class MultiHeadAttention:
    """Multi-head self-attention."""
    
    def __init__(
        self,
        qkv_weight: np.ndarray,
        qkv_bias: np.ndarray,
        out_weight: np.ndarray,
        out_bias: np.ndarray,
        num_heads: int,
        hidden_size: int,
    ):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Split QKV weights
        self.q_weight = qkv_weight[:, :hidden_size]
        self.k_weight = qkv_weight[:, hidden_size:2*hidden_size]
        self.v_weight = qkv_weight[:, 2*hidden_size:]
        
        self.q_bias = qkv_bias[:hidden_size]
        self.k_bias = qkv_bias[hidden_size:2*hidden_size]
        self.v_bias = qkv_bias[2*hidden_size:]
        
        self.out_weight = out_weight
        self.out_bias = out_bias
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = x @ self.q_weight + self.q_bias
        k = x @ self.k_weight + self.k_bias
        v = x @ self.v_weight + self.v_bias
        
        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention scores
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask
        
        # Attention weights
        attn_weights = softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_out = attn_weights @ v
        
        # Reshape back
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Output projection
        out = attn_out @ self.out_weight + self.out_bias
        
        return out


class MLP:
    """Feed-forward network."""
    
    def __init__(
        self,
        fc1_weight: np.ndarray,
        fc1_bias: np.ndarray,
        fc2_weight: np.ndarray,
        fc2_bias: np.ndarray,
    ):
        self.fc1_weight = fc1_weight
        self.fc1_bias = fc1_bias
        self.fc2_weight = fc2_weight
        self.fc2_bias = fc2_bias
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x @ self.fc1_weight + self.fc1_bias
        x = gelu(x)
        x = x @ self.fc2_weight + self.fc2_bias
        return x


class TransformerBlock:
    """Single transformer block."""
    
    def __init__(self, attn: MultiHeadAttention, mlp: MLP, ln1: LayerNorm, ln2: LayerNorm):
        self.attn = attn
        self.mlp = mlp
        self.ln1 = ln1
        self.ln2 = ln2
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Self-attention with residual
        attn_out = self.attn.forward(self.ln1.forward(x))
        x = x + attn_out
        
        # FFN with residual
        mlp_out = self.mlp.forward(self.ln2.forward(x))
        x = x + mlp_out
        
        return x


class NumPyTransformer:
    """NumPy-based transformer for inference."""
    
    def __init__(self, config: Dict[str, Any], weights: Dict[str, np.ndarray]):
        self.config = config
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_hidden_layers']
        self.num_heads = config['num_attention_heads']
        self.max_position_embeddings = config.get('max_position_embeddings', 1024)
        
        # Embeddings
        self.embed_tokens = weights.get('embed_tokens', weights.get('embed_tokens.weight'))
        self.embed_pos = weights.get('embed_pos', weights.get('embed_pos.weight'))
        
        if self.embed_pos is None or self.embed_pos.shape[0] < self.max_position_embeddings:
            # Create positional embeddings if not present
            self.embed_pos = np.random.randn(self.max_position_embeddings, self.hidden_size) * 0.02
        
        # Transformer blocks
        self.blocks = []
        for i in range(self.num_layers):
            prefix = f'layers.{i}'
            
            # Attention
            qkv_weight = weights.get(f'{prefix}.attn.in_proj_weight')
            qkv_bias = weights.get(f'{prefix}.attn.in_proj_bias')
            out_weight = weights.get(f'{prefix}.attn.out_proj.weight')
            out_bias = weights.get(f'{prefix}.attn.out_proj.bias', np.zeros(out_weight.shape[1]))
            
            attn = MultiHeadAttention(
                qkv_weight.T if qkv_weight is not None else None,
                qkv_bias,
                out_weight.T if out_weight is not None else None,
                out_bias,
                self.num_heads,
                self.hidden_size,
            )
            
            # MLP
            fc1_weight = weights.get(f'{prefix}.mlp.0.weight')
            fc1_bias = weights.get(f'{prefix}.mlp.0.bias', np.zeros(fc1_weight.shape[1]))
            fc2_weight = weights.get(f'{prefix}.mlp.3.weight') if f'{prefix}.mlp.3.weight' in weights else weights.get(f'{prefix}.mlp.2.weight')
            fc2_bias = weights.get(f'{prefix}.mlp.3.bias', np.zeros(fc2_weight.shape[1])) if fc2_weight is not None else None
            
            if fc1_weight is not None and fc2_weight is not None:
                mlp = MLP(fc1_weight.T, fc1_bias, fc2_weight.T, fc2_bias)
            else:
                mlp = None
            
            # Layer norms
            ln1_weight = weights.get(f'{prefix}.ln_1.weight', np.ones(self.hidden_size))
            ln1_bias = weights.get(f'{prefix}.ln_1.bias', np.zeros(self.hidden_size))
            ln2_weight = weights.get(f'{prefix}.ln_2.weight', np.ones(self.hidden_size))
            ln2_bias = weights.get(f'{prefix}.ln_2.bias', np.zeros(self.hidden_size))
            
            ln1 = LayerNorm(ln1_weight, ln1_bias)
            ln2 = LayerNorm(ln2_weight, ln2_bias)
            
            self.blocks.append(TransformerBlock(attn, mlp, ln1, ln2))
        
        # Final layer norm
        self.ln_f = LayerNorm(
            weights.get('ln_f.weight', np.ones(self.hidden_size)),
            weights.get('ln_f.bias', np.zeros(self.hidden_size)),
        )
        
        # Language model head
        self.lm_head_weight = weights.get('lm_head.weight', self.embed_tokens)
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Forward pass."""
        
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embed_tokens[input_ids]
        
        # Positional embeddings
        pos_emb = self.embed_pos[:seq_len]
        x = x + pos_emb
        
        # Transformer blocks
        for block in self.blocks:
            if block.attn.q_weight is not None:
                x = block.forward(x)
        
        # Final layer norm
        x = self.ln_f.forward(x)
        
        # Language model head
        logits = x @ self.lm_head_weight.T
        
        return logits
    
    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        """Autoregressive generation."""
        
        for _ in range(max_new_tokens):
            # Truncate if too long
            if input_ids.shape[1] >= self.max_position_embeddings:
                input_ids = input_ids[:, -self.max_position_embeddings:]
            
            # Forward pass
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits = np.sort(next_logits, axis=-1)[:, -top_k:]
                min_top_k = top_k_logits.min(axis=-1, keepdims=True)
                next_logits[next_logits < min_top_k] = -1e9
            
            # Sample
            probs = softmax(next_logits, axis=-1)
            next_token = np.array([np.random.choice(len(p), p=p) for p in probs])
            next_token = next_token.reshape(-1, 1)
            
            # Append
            input_ids = np.concatenate([input_ids, next_token], axis=1)
        
        return input_ids


def load_numpy_weights_from_pt(checkpoint_path: str) -> tuple:
    """Load weights from PyTorch checkpoint (requires torch)."""
    
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required to load .pt checkpoints")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'student_state_dict' in checkpoint:
        state_dict = checkpoint['student_state_dict']
        config = checkpoint.get('student_config')
        if hasattr(config, '__dict__'):
            config = config.__dict__
    else:
        state_dict = checkpoint
        config = None
    
    # Convert to numpy
    weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
    
    return config, weights


def save_numpy_model(model: NumPyTransformer, output_path: str):
    """Save NumPy model to file."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'config': model.config,
        'weights': {k: v.tolist() for k, v in [
            ('embed_tokens', model.embed_tokens),
            ('embed_pos', model.embed_pos),
            ('ln_f.weight', model.ln_f.weight),
            ('ln_f.bias', model.ln_f.bias),
            ('lm_head.weight', model.lm_head_weight),
        ] if v is not None},
    }
    
    # Save block weights
    for i, block in enumerate(model.blocks):
        prefix = f'layers.{i}'
        if block.attn.q_weight is not None:
            data['weights'][f'{prefix}.attn.q_weight'] = block.attn.q_weight.tolist()
            data['weights'][f'{prefix}.attn.k_weight'] = block.attn.k_weight.tolist()
            data['weights'][f'{prefix}.attn.v_weight'] = block.attn.v_weight.tolist()
            data['weights'][f'{prefix}.attn.out_weight'] = block.attn.out_weight.tolist()
        
        if block.mlp is not None:
            data['weights'][f'{prefix}.mlp.fc1_weight'] = block.mlp.fc1_weight.tolist()
            data['weights'][f'{prefix}.mlp.fc2_weight'] = block.mlp.fc2_weight.tolist()
        
        data['weights'][f'{prefix}.ln_1.weight'] = block.ln1.weight.tolist()
        data['weights'][f'{prefix}.ln_1.bias'] = block.ln1.bias.tolist()
        data['weights'][f'{prefix}.ln_2.weight'] = block.ln2.weight.tolist()
        data['weights'][f'{prefix}.ln_2.bias'] = block.ln2.bias.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Saved NumPy model to {output_path}")


def load_numpy_model(model_path: str) -> NumPyTransformer:
    """Load NumPy model from file."""
    
    with open(model_path, 'r') as f:
        data = json.load(f)
    
    config = data['config']
    weights = {k: np.array(v) for k, v in data['weights'].items()}
    
    return NumPyTransformer(config, weights)
