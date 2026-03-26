"""Post-training quantization flow."""

import os
from pathlib import Path
from typing import Optional, Callable
import torch
import onnx
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    QuantType,
    CalibrationDataReader,
)


class SimpleCalibrationDataReader(CalibrationDataReader):
    """Simple calibration data reader for static quantization."""
    
    def __init__(self, data_samples: list, input_name: str):
        self.data_samples = data_samples
        self.input_name = input_name
        self._index = 0
    
    def get_next(self):
        if self._index < len(self.data_samples):
            sample = self.data_samples[self._index]
            self._index += 1
            return {self.input_name: sample}
        return None
    
    def rewind(self):
        self._index = 0


def quantize_dynamic_int8(
    input_path: str,
    output_path: str,
    per_channel: bool = False,
    weight_only: bool = False,
) -> str:
    """Apply dynamic INT8 quantization to ONNX model."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Quantize
    quantize_dynamic(
        input_path,
        str(output_path),
        weight_type=QuantType.QUInt8 if weight_only else QuantType.QInt8,
        per_channel=per_channel,
    )
    
    # Report size reduction
    original_size = os.path.getsize(input_path)
    quantized_size = os.path.getsize(output_path)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Quantized model saved to {output_path}")
    print(f"  - Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"  - Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"  - Reduction: {reduction:.1f}%")
    
    return str(output_path)


def quantize_static_int8(
    input_path: str,
    output_path: str,
    calibration_data: list,
    input_name: str = "input",
    num_calibration_samples: int = 100,
) -> str:
    """Apply static INT8 quantization with calibration."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Limit calibration samples
    calibration_data = calibration_data[:num_calibration_samples]
    
    # Create calibration data reader
    data_reader = SimpleCalibrationDataReader(calibration_data, input_name)
    
    # Quantize
    quantize_static(
        input_path,
        str(output_path),
        data_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
    )
    
    # Report size reduction
    original_size = os.path.getsize(input_path)
    quantized_size = os.path.getsize(output_path)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Static quantized model saved to {output_path}")
    print(f"  - Original size: {original_size / 1024 / 1024:.2f} MB")
    print(f"  - Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
    print(f"  - Reduction: {reduction:.1f}%")
    
    return str(output_path)


def validate_quantized_model(
    original_path: str,
    quantized_path: str,
    test_inputs: list,
    input_name: str = "input",
    atol: float = 1e-2,
) -> tuple:
    """Validate quantized model output."""
    
    import onnxruntime as ort
    
    # Load sessions
    original_session = ort.InferenceSession(original_path)
    quantized_session = ort.InferenceSession(quantized_path)
    
    max_diff = 0.0
    num_samples = 0
    
    for test_input in test_inputs[:10]:  # Test first 10 samples
        original_output = original_session.run(None, {input_name: test_input})[0]
        quantized_output = quantized_session.run(None, {input_name: test_input})[0]
        
        diff = abs(original_output.mean() - quantized_output.mean())
        max_diff = max(max_diff, diff)
        num_samples += 1
    
    avg_diff = max_diff / max(1, num_samples)
    is_valid = avg_diff <= atol
    
    print(f"Quantization validation:")
    print(f"  - Max output difference: {max_diff:.6f}")
    print(f"  - Valid (atol={atol}): {is_valid}")
    
    return is_valid, max_diff


def get_model_size(path: str) -> dict:
    """Get model size information."""
    
    path = Path(path)
    file_size = path.stat().st_size
    
    return {
        'path': str(path),
        'size_bytes': file_size,
        'size_mb': round(file_size / 1024 / 1024, 2),
    }
