// INT8 Quantization for SmLM models
// Supports FP32 -> INT8 conversion and INT8 inference
// Pure C++17 implementation

#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

struct QuantizeConfig {
    bool per_channel = false;  // Use per-channel quantization
    bool symmetric = true;     // Use symmetric quantization
    int num_bits = 8;          // 8-bit quantization
    float clip_ratio = 0.999f; // Clip outliers
    
    size_t quantized_size_bytes(size_t num_elements) const {
        return num_elements * sizeof(int8_t);
    }
    
    size_t scale_size_bytes(size_t num_channels) const {
        if (per_channel) {
            return num_channels * sizeof(float);
        } else {
            return sizeof(float);
        }
    }
};

// Quantization parameters
struct QuantParams {
    float scale;      // Scaling factor
    int8_t zero_point; // Zero point (for asymmetric)
    
    // Quantize FP32 to INT8
    int8_t quantize(float value) const {
        if (symmetric) {
            int32_t q = std::round(value / scale);
            q = std::clamp(q, -127, 127);
            return static_cast<int8_t>(q);
        } else {
            int32_t q = std::round(value / scale) + zero_point;
            q = std::clamp(q, 0, 255);
            return static_cast<int8_t>(q);
        }
    }
    
    // Dequantize INT8 to FP32
    float dequantize(int8_t value) const {
        if (symmetric) {
            return static_cast<float>(value) * scale;
        } else {
            return static_cast<float>(value - zero_point) * scale;
        }
    }
    
    bool symmetric = true;
};

// Quantize a tensor (FP32 -> INT8)
class TensorQuantizer {
private:
    QuantizeConfig config;
    QuantParams params;
    
public:
    TensorQuantizer(const QuantizeConfig& cfg = QuantizeConfig()) : config(cfg) {}
    
    // Compute quantization parameters from data
    QuantParams compute_params(const float* data, size_t size) {
        if (size == 0) {
            return QuantParams{1.0f, 0, true};
        }
        
        // Find min and max
        float min_val = data[0];
        float max_val = data[0];
        
        for (size_t i = 1; i < size; i++) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        
        // Clip outliers
        float range = max_val - min_val;
        if (config.clip_ratio < 1.0f && range > 0) {
            float clip = range * (1.0f - config.clip_ratio) / 2.0f;
            min_val += clip;
            max_val -= clip;
        }
        
        QuantParams p;
        p.symmetric = config.symmetric;
        
        if (config.symmetric) {
            // Symmetric quantization
            float abs_max = std::max(std::abs(min_val), std::abs(max_val));
            p.scale = abs_max / 127.0f;
            p.zero_point = 0;
        } else {
            // Asymmetric quantization
            p.scale = range / 255.0f;
            p.zero_point = static_cast<int8_t>(std::round(-min_val / p.scale));
        }
        
        // Prevent division by zero
        if (p.scale < 1e-10f) p.scale = 1e-10f;
        
        params = p;
        return p;
    }
    
    // Quantize data
    void quantize(const float* input, int8_t* output, size_t size, 
                  const QuantParams& p) {
        for (size_t i = 0; i < size; i++) {
            output[i] = p.quantize(input[i]);
        }
    }
    
    // Dequantize data
    void dequantize(const int8_t* input, float* output, size_t size,
                    const QuantParams& p) {
        for (size_t i = 0; i < size; i++) {
            output[i] = p.dequantize(input[i]);
        }
    }
    
    // Quantize and return params
    QuantParams quantize_with_params(const float* input, int8_t* output, size_t size) {
        QuantParams p = compute_params(input, size);
        quantize(input, output, size, p);
        return p;
    }
    
    // Get last computed params
    const QuantParams& get_params() const { return params; }
};

// INT8 Matrix Multiplication
// C = A * B where A and B are INT8, C is FP32
void matmul_int8(const int8_t* A, const int8_t* B, float* C,
                 int M, int K, int N,
                 float scale_a, float scale_b) {
    float output_scale = scale_a * scale_b;
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            
            for (int k = 0; k < K; k++) {
                acc += static_cast<int32_t>(A[m * K + k]) * 
                       static_cast<int32_t>(B[k * N + n]);
            }
            
            C[m * N + n] = static_cast<float>(acc) * output_scale;
        }
    }
}

// Optimized INT8 GEMM with SIMD-friendly layout
void matmul_int8_optimized(const int8_t* A, const int8_t* B, float* C,
                           int M, int K, int N,
                           float scale_a, float scale_b) {
    float output_scale = scale_a * scale_b;
    
    // Simple optimization: cache-friendly access pattern
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            int8_t a_val = A[m * K + k];
            
            for (int n = 0; n < N; n++) {
                C[m * N + n] += static_cast<float>(a_val * B[k * N + n]) * output_scale;
            }
        }
    }
}

// Quantize entire model weights
struct QuantizedWeights {
    // Quantized data
    std::vector<int8_t> token_embeddings_q;
    std::vector<int8_t> position_embeddings_q;
    
    struct LayerWeights {
        std::vector<int8_t> attention_query_weight_q;
        std::vector<int8_t> attention_key_weight_q;
        std::vector<int8_t> attention_value_weight_q;
        std::vector<int8_t> attention_output_weight_q;
        std::vector<int8_t> mlp_up_weight_q;
        std::vector<int8_t> mlp_down_weight_q;
        
        // Quantization parameters (per-tensor for now)
        QuantParams attention_query_params;
        QuantParams attention_key_params;
        QuantParams attention_value_params;
        QuantParams attention_output_params;
        QuantParams mlp_up_params;
        QuantParams mlp_down_params;
    };
    
    std::vector<LayerWeights> layers;
    
    // Final norm (keep in FP32 for stability)
    std::vector<float> final_norm_weight;
    std::vector<float> final_norm_bias;
    
    // LM head (quantized)
    std::vector<int8_t> lm_head_weight_q;
    QuantParams lm_head_params;
    
    // Config
    QuantizeConfig config;
    
    // Compute compression ratio
    float compression_ratio(const std::vector<float>& original) const {
        size_t original_bytes = original.size() * sizeof(float);
        size_t quantized_bytes = original.size() * sizeof(int8_t);
        return static_cast<float>(quantized_bytes) / static_cast<float>(original_bytes);
    }
};

// Model quantizer
class ModelQuantizer {
private:
    QuantizeConfig config;
    TensorQuantizer tensor_quantizer;
    
public:
    ModelQuantizer(const QuantizeConfig& cfg = QuantizeConfig()) 
        : config(cfg), tensor_quantizer(cfg) {}
    
    // Quantize model weights
    QuantizedWeights quantize(
        const std::vector<float>& token_emb,
        const std::vector<float>& pos_emb,
        const std::vector<std::vector<float>>& layer_weights,
        const std::vector<float>& final_norm_w,
        const std::vector<float>& final_norm_b,
        const std::vector<float>& lm_head
    ) {
        QuantizedWeights qw;
        qw.config = config;
        
        // Quantize embeddings
        qw.token_embeddings_q.resize(token_emb.size());
        tensor_quantizer.quantize_with_params(
            token_emb.data(), qw.token_embeddings_q.data(), token_emb.size());
        
        qw.position_embeddings_q.resize(pos_emb.size());
        tensor_quantizer.quantize_with_params(
            pos_emb.data(), qw.position_embeddings_q.data(), pos_emb.size());
        
        // Quantize layers (simplified - assumes flat weight arrays)
        qw.layers.resize(layer_weights.size());
        for (size_t i = 0; i < layer_weights.size(); i++) {
            auto& lq = qw.layers[i];
            const auto& w = layer_weights[i];
            
            // For demonstration, quantize all layer weights with same params
            lq.attention_query_weight_q.resize(w.size());
            lq.attention_query_params = tensor_quantizer.quantize_with_params(
                w.data(), lq.attention_query_weight_q.data(), w.size());
        }
        
        // Keep final norm in FP32
        qw.final_norm_weight = final_norm_w;
        qw.final_norm_bias = final_norm_b;
        
        // Quantize LM head
        qw.lm_head_weight_q.resize(lm_head.size());
        qw.lm_head_params = tensor_quantizer.quantize_with_params(
            lm_head.data(), qw.lm_head_weight_q.data(), lm_head.size());
        
        // Report compression
        size_t total_original = token_emb.size() + pos_emb.size() + lm_head.size();
        for (const auto& w : layer_weights) {
            total_original += w.size();
        }
        
        size_t total_quantized = qw.token_embeddings_q.size() + 
                                 qw.position_embeddings_q.size() + 
                                 qw.lm_head_weight_q.size();
        for (const auto& l : qw.layers) {
            total_quantized += l.attention_query_weight_q.size();
        }
        
        float ratio = static_cast<float>(total_quantized) / 
                      static_cast<float>(total_original);
        
        std::cout << "Quantization complete:" << std::endl;
        std::cout << "  - Original size: " << (total_original * sizeof(float) / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  - Quantized size: " << (total_quantized * sizeof(int8_t) / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  - Compression ratio: " << (ratio * 100) << "%" << std::endl;
        std::cout << "  - Size reduction: " << ((1.0f - ratio) * 100) << "%" << std::endl;
        
        return qw;
    }
    
    // Save quantized model
    bool save_quantized(const std::string& path, const QuantizedWeights& qw) {
        std::ofstream file(path, std::ios::binary);
        if (!file) return false;
        
        // Magic for quantized model
        file.write("QSLM", 4);  // Quantized SmLM
        
        // Write config
        file.write(reinterpret_cast<const char*>(&qw.config.per_channel), sizeof(bool));
        file.write(reinterpret_cast<const char*>(&qw.config.symmetric), sizeof(bool));
        
        // Helper to write quantized tensor
        auto write_quant_tensor = [&file](const std::vector<int8_t>& data, 
                                          const QuantParams& params) {
            uint32_t size = data.size();
            file.write(reinterpret_cast<const char*>(&size), sizeof(size));
            file.write(reinterpret_cast<const char*>(data.data()), size * sizeof(int8_t));
            file.write(reinterpret_cast<const char*>(&params.scale), sizeof(float));
            file.write(reinterpret_cast<const char*>(&params.zero_point), sizeof(int8_t));
            file.write(reinterpret_cast<const char*>(&params.symmetric), sizeof(bool));
        };
        
        // Write embeddings
        write_quant_tensor(qw.token_embeddings_q, qw.layers.empty() ? QuantParams{} : qw.layers[0].attention_query_params);
        write_quant_tensor(qw.position_embeddings_q, qw.layers.empty() ? QuantParams{} : qw.layers[0].attention_query_params);
        
        // Write layers
        uint32_t num_layers = qw.layers.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
        
        for (const auto& layer : qw.layers) {
            write_quant_tensor(layer.attention_query_weight_q, layer.attention_query_params);
        }
        
        // Write final norm (FP32)
        uint32_t fn_size = qw.final_norm_weight.size();
        file.write(reinterpret_cast<const char*>(&fn_size), sizeof(fn_size));
        file.write(reinterpret_cast<const char*>(qw.final_norm_weight.data()), 
                   fn_size * sizeof(float));
        
        // Write LM head
        write_quant_tensor(qw.lm_head_weight_q, qw.lm_head_params);
        
        file.close();
        std::cout << "Quantized model saved to: " << path << std::endl;
        return true;
    }
};

#endif // QUANTIZE_H
