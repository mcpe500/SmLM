"""ONNX export flow."""

import os
import torch
import onnx
from pathlib import Path
from typing import Optional, Tuple
from .student import StudentTransformer, StudentConfig


def export_to_onnx(
    model: StudentTransformer,
    config: StudentConfig,
    output_path: str,
    seq_len: int = 512,
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> str:
    """Export student model to ONNX format."""
    
    model.eval()
    device = next(model.parameters()).device
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input
    batch_size = 1
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    
    # Define dynamic axes
    axes = {}
    if dynamic_axes:
        axes = {
            'input': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size', 1: 'sequence'},
        }
    
    # Export
    torch.onnx.export(
        model,
        input_ids,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=axes if dynamic_axes else None,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
    )
    
    # Validate export
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    print(f"Exported model to {output_path}")
    print(f"  - Input shape: {list(input_ids.shape)}")
    print(f"  - Output shape: [batch, seq, {config.vocab_size}]")
    
    return str(output_path)


def validate_onnx_output(
    pytorch_model: StudentTransformer,
    onnx_path: str,
    test_inputs: Optional[torch.Tensor] = None,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Tuple[bool, float]:
    """Validate ONNX output matches PyTorch output."""
    
    import onnxruntime as ort
    
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    
    # Create test input if not provided
    if test_inputs is None:
        test_inputs = torch.randint(0, 1000, (1, 64), dtype=torch.long, device=device)
    
    # PyTorch output
    with torch.no_grad():
        pt_output = pytorch_model(test_inputs).cpu().numpy()
    
    # ONNX output
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, sess_options)
    input_name = session.get_inputs()[0].name
    
    onnx_output = session.run(None, {input_name: test_inputs.cpu().numpy()})[0]
    
    # Compare
    max_diff = float(torch.abs(torch.tensor(pt_output) - torch.tensor(onnx_output)).max())
    is_valid = max_diff <= atol + rtol * torch.tensor(onnx_output).abs().max()
    
    print(f"ONNX validation:")
    print(f"  - Max absolute difference: {max_diff:.6f}")
    print(f"  - Valid: {is_valid}")
    
    return is_valid, max_diff


def optimize_onnx(
    input_path: str,
    output_path: str,
    optimization_level: str = "all",
) -> str:
    """Optimize ONNX model for inference."""
    
    from onnxruntime.transformers.optimizer import optimize_model
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Optimize
    optimized_model = optimize_model(
        input_path,
        optimization_level=optimization_level,
        num_heads=0,  # Auto-detect
        hidden_size=0,  # Auto-detect
    )
    
    optimized_model.save_model_to_file(str(output_path))
    
    print(f"Optimized model saved to {output_path}")
    
    return str(output_path)


def get_onnx_info(onnx_path: str) -> dict:
    """Get information about ONNX model."""
    
    import onnx
    
    model = onnx.load(onnx_path)
    
    # Get input/output shapes
    inputs = [(inp.name, list(inp.type.tensor_type.shape.dim)) for inp in model.graph.input]
    outputs = [(out.name, list(out.type.tensor_type.shape.dim)) for out in model.graph.output]
    
    # Count parameters (approximate)
    param_count = 0
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                param_count += sum(attr.t.dims)
    
    # File size
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    
    return {
        'path': onnx_path,
        'file_size_mb': round(file_size_mb, 2),
        'inputs': inputs,
        'outputs': outputs,
        'opset_version': model.opset_import[0].version,
    }
