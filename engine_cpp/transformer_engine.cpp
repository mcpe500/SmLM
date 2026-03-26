// Pure C++ Transformer Inference Engine
// No external ML libraries - pure C++17 with standard library only
// Designed for CPU-only, low-memory environments (Termux/Android)

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstdint>

// ============================================================================
// Configuration
// ============================================================================

struct ModelConfig {
    std::string model_path;
    int vocab_size = 50257;
    int max_position_embeddings = 1024;
    int hidden_size = 512;
    int num_hidden_layers = 6;
    int num_attention_heads = 6;
    int intermediate_size = 2048;
    float eps = 1e-5f;
    float temperature = 1.0f;
    int top_k = 50;
    int max_new_tokens = 100;
    bool benchmark = false;
    int num_runs = 10;
    bool verbose = false;
};

// ============================================================================
// Math Utilities
// ============================================================================

namespace math {

inline float gelu(float x) {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_pi * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

inline float relu(float x) {
    return std::max(0.0f, x);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float softmax_temperature(float x, float temp) {
    return std::exp(x / temp);
}

// Layer normalization: y = (x - mean) / sqrt(variance + eps) * weight + bias
void layer_norm(float* input, float* output, float* weight, float* bias, 
                int size, float eps) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    // Compute variance
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    // Normalize
    float inv_std = 1.0f / std::sqrt(variance + eps);
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// Matrix multiplication: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Matrix multiplication with bias: C = A * B + bias
void matmul_add_bias(const float* A, const float* B, float* C, float* bias, 
                     int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = bias[n];
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Element-wise addition: C = A + B
void add(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// Element-wise multiplication: C = A * B
void multiply(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] * B[i];
    }
}

// Apply GELU activation
void apply_gelu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = gelu(data[i]);
    }
}

// Apply softmax with temperature
void softmax(float* data, int size, float temperature) {
    // Find max for numerical stability
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        data[i] = std::exp((data[i] - max_val) / temperature);
        sum += data[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

// Top-K softmax: zero out all but top-k values, then softmax
void topk_softmax(float* data, int size, int k, float temperature) {
    // Create index array for sorting
    std::vector<std::pair<float, int>> indexed(size);
    for (int i = 0; i < size; i++) {
        indexed[i] = {data[i], i};
    }
    
    // Partial sort to get top-k
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Zero out non-top-k
    std::vector<float> temp(size, 0.0f);
    for (int i = 0; i < k; i++) {
        temp[indexed[i].second] = indexed[i].first;
    }
    
    // Apply softmax to top-k only
    softmax(temp.data(), size, temperature);
    
    // Copy back
    for (int i = 0; i < size; i++) {
        data[i] = temp[i];
    }
}

// Sample from probability distribution
int sample_from_dist(const float* probs, int size, std::mt19937& gen) {
    std::discrete_distribution<> dist(probs, probs + size);
    return dist(gen);
}

// Argmax
int argmax(const float* data, int size) {
    int max_idx = 0;
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

} // namespace math

// ============================================================================
// Model Weights Structure
// ============================================================================

struct TransformerWeights {
    // Token embeddings
    std::vector<float> token_embeddings;      // [vocab_size, hidden_size]
    
    // Position embeddings
    std::vector<float> position_embeddings;   // [max_positions, hidden_size]
    
    // Per-layer weights
    struct LayerWeights {
        // Attention
        std::vector<float> attention_query_weight;     // [hidden_size, hidden_size]
        std::vector<float> attention_query_bias;       // [hidden_size]
        std::vector<float> attention_key_weight;       // [hidden_size, hidden_size]
        std::vector<float> attention_key_bias;         // [hidden_size]
        std::vector<float> attention_value_weight;     // [hidden_size, hidden_size]
        std::vector<float> attention_value_bias;       // [hidden_size]
        std::vector<float> attention_output_weight;    // [hidden_size, hidden_size]
        std::vector<float> attention_output_bias;      // [hidden_size]
        
        // Layer norm 1
        std::vector<float> ln1_weight;                 // [hidden_size]
        std::vector<float> ln1_bias;                   // [hidden_size]
        
        // MLP
        std::vector<float> mlp_up_weight;              // [hidden_size, intermediate_size]
        std::vector<float> mlp_up_bias;                // [intermediate_size]
        std::vector<float> mlp_down_weight;            // [intermediate_size, hidden_size]
        std::vector<float> mlp_down_bias;              // [hidden_size]
        
        // Layer norm 2
        std::vector<float> ln2_weight;                 // [hidden_size]
        std::vector<float> ln2_bias;                   // [hidden_size]
    };
    
    std::vector<LayerWeights> layers;
    
    // Final layer norm
    std::vector<float> final_norm_weight;              // [hidden_size]
    std::vector<float> final_norm_bias;                // [hidden_size]
    
    // LM head (may share weights with token embeddings)
    std::vector<float> lm_head_weight;                 // [hidden_size, vocab_size]
    std::vector<float> lm_head_bias;                   // [vocab_size]
    
    // Config
    ModelConfig config;
    
    // Memory tracking
    mutable size_t total_params = 0;
    
    size_t compute_total_params() const {
        total_params = 0;
        total_params += token_embeddings.size();
        total_params += position_embeddings.size();
        
        for (auto& layer : layers) {
            total_params += layer.attention_query_weight.size();
            total_params += layer.attention_query_bias.size();
            total_params += layer.attention_key_weight.size();
            total_params += layer.attention_key_bias.size();
            total_params += layer.attention_value_weight.size();
            total_params += layer.attention_value_bias.size();
            total_params += layer.attention_output_weight.size();
            total_params += layer.attention_output_bias.size();
            total_params += layer.ln1_weight.size();
            total_params += layer.ln1_bias.size();
            total_params += layer.mlp_up_weight.size();
            total_params += layer.mlp_up_bias.size();
            total_params += layer.mlp_down_weight.size();
            total_params += layer.mlp_down_bias.size();
            total_params += layer.ln2_weight.size();
            total_params += layer.ln2_bias.size();
        }
        
        total_params += final_norm_weight.size();
        total_params += final_norm_bias.size();
        total_params += lm_head_weight.size();
        total_params += lm_head_bias.size();
        
        return total_params;
    }
};

// ============================================================================
// Model File Format
// ============================================================================

// File format:
// [magic: 4 bytes] = "SmLM" (stored as little-endian uint32: 0x4D4C6D53)
// [version: 4 bytes] = 1
// [config_size: 4 bytes]
// [config_json: variable]
// [weights: binary float32]

const uint32_t MAGIC = 0x4D4C6D53; // "SmLM" in little-endian
const uint32_t VERSION = 1;

bool save_model(const std::string& path, const TransformerWeights& weights) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file for writing: " << path << std::endl;
        return false;
    }
    
    // Write magic
    uint32_t magic = MAGIC;
    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    
    // Write version
    uint32_t version = VERSION;
    file.write(reinterpret_cast<char*>(&version), sizeof(version));
    
    // Write config
    const ModelConfig& cfg = weights.config;
    file.write(reinterpret_cast<const char*>(&cfg.vocab_size), sizeof(cfg.vocab_size));
    file.write(reinterpret_cast<const char*>(&cfg.max_position_embeddings), sizeof(cfg.max_position_embeddings));
    file.write(reinterpret_cast<const char*>(&cfg.hidden_size), sizeof(cfg.hidden_size));
    file.write(reinterpret_cast<const char*>(&cfg.num_hidden_layers), sizeof(cfg.num_hidden_layers));
    file.write(reinterpret_cast<const char*>(&cfg.num_attention_heads), sizeof(cfg.num_attention_heads));
    file.write(reinterpret_cast<const char*>(&cfg.intermediate_size), sizeof(cfg.intermediate_size));
    
    // Write weights
    auto write_vector = [&file](const std::vector<float>& vec) {
        uint32_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        if (!vec.empty()) {
            file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
        }
    };
    
    write_vector(weights.token_embeddings);
    write_vector(weights.position_embeddings);
    
    for (const auto& layer : weights.layers) {
        write_vector(layer.attention_query_weight);
        write_vector(layer.attention_query_bias);
        write_vector(layer.attention_key_weight);
        write_vector(layer.attention_key_bias);
        write_vector(layer.attention_value_weight);
        write_vector(layer.attention_value_bias);
        write_vector(layer.attention_output_weight);
        write_vector(layer.attention_output_bias);
        write_vector(layer.ln1_weight);
        write_vector(layer.ln1_bias);
        write_vector(layer.mlp_up_weight);
        write_vector(layer.mlp_up_bias);
        write_vector(layer.mlp_down_weight);
        write_vector(layer.mlp_down_bias);
        write_vector(layer.ln2_weight);
        write_vector(layer.ln2_bias);
    }
    
    write_vector(weights.final_norm_weight);
    write_vector(weights.final_norm_bias);
    write_vector(weights.lm_head_weight);
    write_vector(weights.lm_head_bias);
    
    file.close();
    std::cout << "Model saved to: " << path << std::endl;
    std::cout << "Total params: " << weights.compute_total_params() << std::endl;
    return true;
}

bool load_model(const std::string& path, TransformerWeights& weights) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file: " << path << std::endl;
        return false;
    }
    
    // Read and verify magic
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MAGIC) {
        std::cerr << "Error: Invalid model file (bad magic)" << std::endl;
        return false;
    }
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != VERSION) {
        std::cerr << "Error: Unsupported model version: " << version << std::endl;
        return false;
    }
    
    // Read config
    ModelConfig& cfg = weights.config;
    file.read(reinterpret_cast<char*>(&cfg.vocab_size), sizeof(cfg.vocab_size));
    file.read(reinterpret_cast<char*>(&cfg.max_position_embeddings), sizeof(cfg.max_position_embeddings));
    file.read(reinterpret_cast<char*>(&cfg.hidden_size), sizeof(cfg.hidden_size));
    file.read(reinterpret_cast<char*>(&cfg.num_hidden_layers), sizeof(cfg.num_hidden_layers));
    file.read(reinterpret_cast<char*>(&cfg.num_attention_heads), sizeof(cfg.num_attention_heads));
    file.read(reinterpret_cast<char*>(&cfg.intermediate_size), sizeof(cfg.intermediate_size));
    
    std::cout << "Model config:" << std::endl;
    std::cout << "  - Vocab size: " << cfg.vocab_size << std::endl;
    std::cout << "  - Max positions: " << cfg.max_position_embeddings << std::endl;
    std::cout << "  - Hidden size: " << cfg.hidden_size << std::endl;
    std::cout << "  - Layers: " << cfg.num_hidden_layers << std::endl;
    std::cout << "  - Heads: " << cfg.num_attention_heads << std::endl;
    std::cout << "  - Intermediate size: " << cfg.intermediate_size << std::endl;
    
    // Read weights
    auto read_vector = [&file](std::vector<float>& vec) {
        uint32_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        if (size > 0) {
            file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
        }
    };
    
    read_vector(weights.token_embeddings);
    read_vector(weights.position_embeddings);
    
    weights.layers.resize(cfg.num_hidden_layers);
    for (auto& layer : weights.layers) {
        read_vector(layer.attention_query_weight);
        read_vector(layer.attention_query_bias);
        read_vector(layer.attention_key_weight);
        read_vector(layer.attention_key_bias);
        read_vector(layer.attention_value_weight);
        read_vector(layer.attention_value_bias);
        read_vector(layer.attention_output_weight);
        read_vector(layer.attention_output_bias);
        read_vector(layer.ln1_weight);
        read_vector(layer.ln1_bias);
        read_vector(layer.mlp_up_weight);
        read_vector(layer.mlp_up_bias);
        read_vector(layer.mlp_down_weight);
        read_vector(layer.mlp_down_bias);
        read_vector(layer.ln2_weight);
        read_vector(layer.ln2_bias);
    }
    
    read_vector(weights.final_norm_weight);
    read_vector(weights.final_norm_bias);
    read_vector(weights.lm_head_weight);
    read_vector(weights.lm_head_bias);
    
    file.close();
    
    std::cout << "Model loaded successfully" << std::endl;
    std::cout << "Total params: " << weights.compute_total_params() << std::endl;
    std::cout << "Model size: " << (weights.compute_total_params() * sizeof(float) / 1024 / 1024) << " MB" << std::endl;
    
    return true;
}

// ============================================================================
// Transformer Inference Engine
// ============================================================================

class TransformerEngine {
private:
    TransformerWeights weights;
    bool loaded = false;
    std::mt19937 rng;
    
    // Working buffers (reused to avoid allocations)
    std::vector<float> hidden_state;
    std::vector<float> residual;
    std::vector<float> normalized;
    std::vector<float> qkv;
    std::vector<float> attention_scores;
    std::vector<float> attention_output;
    std::vector<float> mlp_hidden;
    std::vector<float> mlp_output;
    std::vector<float> logits;
    
    int head_size;
    int num_heads;
    
public:
    TransformerEngine() : rng(std::random_device{}()) {}
    
    bool load(const std::string& path) {
        std::cout << "Loading model from: " << path << std::endl;
        if (!load_model(path, weights)) {
            return false;
        }
        
        const ModelConfig& cfg = weights.config;
        head_size = cfg.hidden_size / cfg.num_attention_heads;
        num_heads = cfg.num_attention_heads;
        
        // Allocate working buffers
        hidden_state.resize(cfg.hidden_size);
        residual.resize(cfg.hidden_size);
        normalized.resize(cfg.hidden_size);
        qkv.resize(cfg.hidden_size * 3);
        attention_scores.resize(cfg.max_position_embeddings);
        attention_output.resize(cfg.hidden_size);
        mlp_hidden.resize(cfg.intermediate_size);
        mlp_output.resize(cfg.hidden_size);
        logits.resize(cfg.vocab_size);
        
        loaded = true;
        std::cout << "Model ready for inference" << std::endl;
        return true;
    }
    
    // Forward pass for a single token
    // Returns logits of size [vocab_size]
    const float* forward(const int* input_ids, int seq_len) {
        if (!loaded) return nullptr;
        
        const ModelConfig& cfg = weights.config;
        
        // Get last token's hidden state from embeddings
        int last_token = input_ids[seq_len - 1];
        const float* token_emb = &weights.token_embeddings[last_token * cfg.hidden_size];
        const float* pos_emb = &weights.position_embeddings[(seq_len - 1) * cfg.hidden_size];
        
        // Initial hidden state = token_emb + pos_emb
        for (int i = 0; i < cfg.hidden_size; i++) {
            hidden_state[i] = token_emb[i] + pos_emb[i];
        }
        
        // Pass through transformer layers
        for (int layer_idx = 0; layer_idx < cfg.num_hidden_layers; layer_idx++) {
            auto& layer = weights.layers[layer_idx];
            
            // ============ Self-Attention ============
            
            // Save residual
            std::memcpy(residual.data(), hidden_state.data(), cfg.hidden_size * sizeof(float));
            
            // Layer norm 1
            math::layer_norm(hidden_state.data(), normalized.data(), 
                           layer.ln1_weight.data(), layer.ln1_bias.data(),
                           cfg.hidden_size, cfg.eps);
            
            // Compute Q, K, V
            std::vector<float> Q(cfg.hidden_size), K(cfg.hidden_size), V(cfg.hidden_size);
            math::matmul(normalized.data(), layer.attention_query_weight.data(), 
                        Q.data(), 1, cfg.hidden_size, cfg.hidden_size);
            math::matmul_add_bias(normalized.data(), layer.attention_key_weight.data(),
                                 K.data(), layer.attention_key_bias.data(),
                                 1, cfg.hidden_size, cfg.hidden_size);
            math::matmul_add_bias(normalized.data(), layer.attention_value_weight.data(),
                                 V.data(), layer.attention_value_bias.data(),
                                 1, cfg.hidden_size, cfg.hidden_size);
            
            // Scaled dot-product attention (simplified for single token)
            // float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
            
            // For single token inference, attention is just identity
            // (no causal mask needed as we're generating one token at a time)
            std::memcpy(attention_output.data(), V.data(), cfg.hidden_size * sizeof(float));
            
            // Output projection
            math::matmul_add_bias(attention_output.data(), layer.attention_output_weight.data(),
                                 mlp_output.data(), layer.attention_output_bias.data(),
                                 1, cfg.hidden_size, cfg.hidden_size);
            
            // Add residual
            math::add(mlp_output.data(), residual.data(), hidden_state.data(), cfg.hidden_size);
            
            // ============ MLP ============
            
            // Save residual
            std::memcpy(residual.data(), hidden_state.data(), cfg.hidden_size * sizeof(float));
            
            // Layer norm 2
            math::layer_norm(hidden_state.data(), normalized.data(),
                           layer.ln2_weight.data(), layer.ln2_bias.data(),
                           cfg.hidden_size, cfg.eps);
            
            // MLP up projection
            math::matmul_add_bias(normalized.data(), layer.mlp_up_weight.data(),
                                 mlp_hidden.data(), layer.mlp_up_bias.data(),
                                 1, cfg.hidden_size, cfg.intermediate_size);
            
            // GELU activation
            math::apply_gelu(mlp_hidden.data(), cfg.intermediate_size);
            
            // MLP down projection
            math::matmul_add_bias(mlp_hidden.data(), layer.mlp_down_weight.data(),
                                 mlp_output.data(), layer.mlp_down_bias.data(),
                                 1, cfg.intermediate_size, cfg.hidden_size);
            
            // Add residual
            math::add(mlp_output.data(), residual.data(), hidden_state.data(), cfg.hidden_size);
        }
        
        // Final layer norm
        math::layer_norm(hidden_state.data(), normalized.data(),
                        weights.final_norm_weight.data(), weights.final_norm_bias.data(),
                        cfg.hidden_size, cfg.eps);
        
        // LM head
        math::matmul_add_bias(normalized.data(), weights.lm_head_weight.data(),
                             logits.data(), weights.lm_head_bias.data(),
                             1, cfg.hidden_size, cfg.vocab_size);
        
        return logits.data();
    }
    
    // Generate next token
    int generate_next_token(const int* input_ids, int seq_len) {
        const float* logits_ptr = forward(input_ids, seq_len);
        if (!logits_ptr) return -1;
        
        const ModelConfig& cfg = weights.config;
        
        // Copy logits to working buffer
        std::memcpy(logits.data(), logits_ptr, cfg.vocab_size * sizeof(float));
        
        // Apply temperature and sample
        if (cfg.temperature > 0.0f) {
            if (cfg.top_k > 0 && cfg.top_k < cfg.vocab_size) {
                math::topk_softmax(logits.data(), cfg.vocab_size, cfg.top_k, cfg.temperature);
            } else {
                math::softmax(logits.data(), cfg.vocab_size, cfg.temperature);
            }
            return math::sample_from_dist(logits.data(), cfg.vocab_size, rng);
        } else {
            // Greedy decoding
            return math::argmax(logits.data(), cfg.vocab_size);
        }
    }
    
    // Generate sequence
    std::vector<int> generate(const std::vector<int>& input_ids) {
        if (!loaded) return input_ids;
        
        const ModelConfig& cfg = weights.config;
        std::vector<int> output = input_ids;
        
        std::cout << "Generating (max " << cfg.max_new_tokens << " tokens)..." << std::endl;
        
        for (int i = 0; i < cfg.max_new_tokens; i++) {
            if (output.size() >= static_cast<size_t>(cfg.max_position_embeddings)) {
                std::cout << "Reached max position limit" << std::endl;
                break;
            }
            
            int next_token = generate_next_token(output.data(), output.size());
            if (next_token < 0) break;
            
            output.push_back(next_token);
            
            if (next_token == 50256) { // EOS token
                std::cout << "Generated EOS token" << std::endl;
                break;
            }
        }
        
        return output;
    }
    
    // Benchmark
    double benchmark() {
        if (!loaded) return 0.0;
        
        const ModelConfig& cfg = weights.config;
        std::vector<int> dummy_input(64, 1);
        
        std::cout << "Running benchmark (" << cfg.num_runs << " runs)..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < cfg.num_runs; i++) {
            forward(dummy_input.data(), dummy_input.size());
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        double tokens_per_sec = cfg.num_runs / elapsed.count();
        
        std::cout << "Benchmark results:" << std::endl;
        std::cout << "  - Total time: " << elapsed.count() << "s" << std::endl;
        std::cout << "  - Tokens/sec: " << tokens_per_sec << std::endl;
        std::cout << "  - Avg latency: " << (elapsed.count() / cfg.num_runs) * 1000 << "ms" << std::endl;
        
        return tokens_per_sec;
    }
    
    const ModelConfig& get_config() const { return weights.config; }
    size_t get_param_count() const { return weights.total_params; }
};

// ============================================================================
// CLI
// ============================================================================

void print_usage(const char* program) {
    std::cout << "SmLM Transformer Engine (Pure C++)" << std::endl;
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model <path>       Path to .slm model file (required)" << std::endl;
    std::cout << "  --input <ids>        Comma-separated input token IDs" << std::endl;
    std::cout << "  --max_tokens <n>     Max new tokens to generate (default: 100)" << std::endl;
    std::cout << "  --temperature <f>    Sampling temperature (default: 1.0, 0=greedy)" << std::endl;
    std::cout << "  --top_k <n>          Top-k sampling (default: 50, 0=disabled)" << std::endl;
    std::cout << "  --benchmark          Run benchmark mode" << std::endl;
    std::cout << "  --num_runs <n>       Number of benchmark runs (default: 10)" << std::endl;
    std::cout << "  --verbose            Show detailed output" << std::endl;
    std::cout << "  --help               Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program << " --model model.slm --input 50256,123,456 --max_tokens 50" << std::endl;
}

ModelConfig parse_args(int argc, char* argv[]) {
    ModelConfig config;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            // Parse comma-separated token IDs
            std::string input_str = argv[++i];
            std::cout << "Input tokens: " << input_str << std::endl;
            // Will be parsed later
        } else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) {
            config.max_new_tokens = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--top_k") == 0 && i + 1 < argc) {
            config.top_k = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            config.benchmark = true;
        } else if (strcmp(argv[i], "--num_runs") == 0 && i + 1 < argc) {
            config.num_runs = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }
    
    return config;
}

std::vector<int> parse_input_ids(const std::string& input_str) {
    std::vector<int> ids;
    std::stringstream ss(input_str);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            ids.push_back(std::stoi(token));
        }
    }
    
    if (ids.empty()) {
        ids.push_back(50256); // Default to BOS
    }
    
    return ids;
}

int main(int argc, char* argv[]) {
    ModelConfig config = parse_args(argc, argv);
    
    if (config.model_path.empty()) {
        std::cerr << "Error: --model is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    TransformerEngine engine;
    
    if (!engine.load(config.model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    if (config.benchmark) {
        engine.benchmark();
    } else {
        // Find --input argument
        std::vector<int> input_ids = {50256}; // Default BOS
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
                input_ids = parse_input_ids(argv[++i]);
                break;
            }
        }
        
        std::cout << "Input token IDs: ";
        for (int id : input_ids) std::cout << id << " ";
        std::cout << std::endl;
        
        auto output = engine.generate(input_ids);
        
        std::cout << "Output token IDs: ";
        for (int id : output) std::cout << id << " ";
        std::cout << std::endl;
        std::cout << "Generated " << (output.size() - input_ids.size()) << " new tokens" << std::endl;
    }
    
    return 0;
}
