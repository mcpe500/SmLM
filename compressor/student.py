"""Student model architecture generator."""

from dataclasses import dataclass
from typing import Optional, Any
import torch
import torch.nn as nn

# Optional transformers import
try:
    from transformers import PretrainedConfig, PreTrainedModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PretrainedConfig = object
    PreTrainedModel = torch.nn.Module


@dataclass
class StudentConfig:
    """Configuration for a dense student model."""
    
    # Base architecture (must match teacher family)
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    
    # Architecture hyperparameters
    num_hidden_layers: int = 6  # Reduced from teacher
    num_attention_heads: int = 6  # Reduced from teacher
    hidden_size: int = 512  # Reduced from teacher
    intermediate_size: int = 2048  # FFN size, reduced from teacher
    
    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Activation
    hidden_act: str = "gelu"
    
    @classmethod
    def from_teacher_config(
        cls,
        teacher_config: PretrainedConfig,
        shrink_factor: float = 0.5,
        layer_ratio: float = 0.5,
        head_ratio: float = 1.0,
        ffn_ratio: float = 0.5,
    ) -> "StudentConfig":
        """Create student config by shrinking teacher config."""
        
        # Calculate student dimensions
        num_hidden_layers = max(2, int(teacher_config.num_hidden_layers * layer_ratio))
        hidden_size = int(teacher_config.hidden_size * shrink_factor)
        
        # Ensure hidden_size is divisible by num_attention_heads
        num_attention_heads = teacher_config.num_attention_heads
        if head_ratio < 1.0:
            num_attention_heads = max(1, int(num_attention_heads * head_ratio))
        
        # Adjust hidden_size to be divisible by num_attention_heads
        while hidden_size % num_attention_heads != 0:
            hidden_size -= 1
        hidden_size = max(64, hidden_size)
        
        # FFN size
        if hasattr(teacher_config, 'intermediate_size'):
            intermediate_size = int(teacher_config.intermediate_size * ffn_ratio)
        else:
            intermediate_size = int(teacher_config.hidden_size * 4 * ffn_ratio)
        
        # Ensure intermediate_size is divisible by hidden_size for some architectures
        intermediate_size = max(intermediate_size, hidden_size)
        
        return cls(
            vocab_size=teacher_config.vocab_size,
            max_position_embeddings=teacher_config.max_position_embeddings,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=teacher_config.hidden_dropout_prob,
            attention_probs_dropout_prob=teacher_config.attention_probs_dropout_prob,
            hidden_act=getattr(teacher_config, 'hidden_act', 'gelu'),
        )


class StudentTransformer(nn.Module):
    """Simple transformer student model."""
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_pos = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        tok_emb = self.embed_tokens(input_ids)
        pos_emb = self.embed_pos(torch.arange(seq_len, device=input_ids.device))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate if too long
            if input_ids.shape[1] >= self.config.max_position_embeddings:
                input_ids = input_ids[:, -self.config.max_position_embeddings:]
            
            # Forward pass
            logits = self.forward(input_ids)
            next_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, config: StudentConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention
        attn_out, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + attn_out
        
        # FFN
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        
        return x


def create_student_from_teacher(
    teacher_model: PreTrainedModel,
    shrink_factor: float = 0.5,
    layer_ratio: float = 0.5,
) -> StudentTransformer:
    """Create a student model from a teacher model."""
    
    teacher_config = teacher_model.config
    student_config = StudentConfig.from_teacher_config(
        teacher_config,
        shrink_factor=shrink_factor,
        layer_ratio=layer_ratio,
    )
    
    student = StudentTransformer(student_config)
    
    return student, student_config
