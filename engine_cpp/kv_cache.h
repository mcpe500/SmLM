// KV Cache implementation for faster transformer inference
// Caches K,V projections to avoid recomputing previous tokens
// Pure C++17 implementation

#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <vector>
#include <cstdint>

struct KVCacheConfig {
    int max_batch_size = 1;
    int max_seq_len = 1024;
    int num_layers = 6;
    int num_heads = 6;
    int head_dim = 64;  // hidden_size / num_heads
    
    size_t cache_size_bytes() const {
        // 2 for K and V, 2 for each layer
        size_t per_layer = 2 * max_batch_size * max_seq_len * num_heads * head_dim * sizeof(float);
        return per_layer * num_layers;
    }
};

class KVCache {
private:
    KVCacheConfig config;
    
    // Cache layout: [layer][batch, seq, head, dim]
    // Flattened for cache efficiency
    std::vector<float> k_cache;  // Key cache
    std::vector<float> v_cache;  // Value cache
    
    // Current sequence lengths per batch item
    std::vector<int> seq_lengths;
    
    bool initialized = false;
    
public:
    KVCache(const KVCacheConfig& cfg) : config(cfg) {
        seq_lengths.resize(config.max_batch_size, 0);
    }
    
    // Initialize cache buffers
    void init() {
        size_t cache_size = config.max_batch_size * config.max_seq_len * 
                           config.num_heads * config.head_dim;
        
        k_cache.resize(cache_size * config.num_layers * 2, 0.0f);
        v_cache.resize(cache_size * config.num_layers * 2, 0.0f);
        
        initialized = true;
    }
    
    // Reset cache for new sequence
    void reset() {
        if (!initialized) init();
        std::fill(seq_lengths.begin(), seq_lengths.end(), 0);
    }
    
    // Reset cache for specific batch item
    void reset_batch(int batch_idx) {
        if (batch_idx >= 0 && batch_idx < config.max_batch_size) {
            seq_lengths[batch_idx] = 0;
        }
    }
    
    // Get current sequence length for batch item
    int get_seq_length(int batch_idx = 0) const {
        if (batch_idx >= 0 && batch_idx < config.max_batch_size) {
            return seq_lengths[batch_idx];
        }
        return 0;
    }
    
    // Store K,V for current token
    // key: [num_heads, head_dim]
    // value: [num_heads, head_dim]
    void store(int layer, int batch_idx, int position, 
               const float* key, const float* value) {
        if (!initialized) init();
        if (layer >= config.num_layers) return;
        if (batch_idx >= config.max_batch_size) return;
        if (position >= config.max_seq_len) return;
        
        size_t head_dim = config.head_dim;
        size_t num_heads = config.num_heads;
        
        // Calculate offset
        size_t layer_offset = layer * config.max_batch_size * config.max_seq_len * num_heads * head_dim;
        size_t batch_offset = batch_idx * config.max_seq_len * num_heads * head_dim;
        size_t pos_offset = position * num_heads * head_dim;
        size_t base_offset = layer_offset + batch_offset + pos_offset;
        
        // Store K and V
        for (size_t h = 0; h < num_heads; h++) {
            size_t h_offset = h * head_dim;
            size_t k_offset = base_offset + h_offset;
            size_t v_offset = k_offset + config.max_batch_size * config.max_seq_len * num_heads * head_dim;
            
            for (size_t d = 0; d < head_dim; d++) {
                k_cache[k_offset + d] = key[h * head_dim + d];
                v_cache[v_offset + d] = value[h * head_dim + d];
            }
        }
        
        // Update sequence length
        if (position >= seq_lengths[batch_idx]) {
            seq_lengths[batch_idx] = position + 1;
        }
    }
    
    // Retrieve all cached K,V up to current position
    // Returns pointers to cache data
    void retrieve(int layer, int batch_idx, int current_pos,
                  float** k_out, float** v_out, int* seq_len_out) {
        if (!initialized) init();
        
        int seq_len = seq_lengths[batch_idx];
        if (seq_len_out) *seq_len_out = seq_len;
        
        size_t head_dim = config.head_dim;
        size_t num_heads = config.num_heads;
        
        // Calculate offset
        size_t layer_offset = layer * config.max_batch_size * config.max_seq_len * num_heads * head_dim;
        size_t batch_offset = batch_idx * config.max_seq_len * num_heads * head_dim;
        size_t base_offset = layer_offset + batch_offset;
        
        // Return pointers to start of cache for this layer/batch
        // Caller should only use up to current_pos
        static std::vector<float> k_temp, v_temp;
        
        if (current_pos == 0) {
            // First token - no cache yet
            k_temp.clear();
            v_temp.clear();
            if (k_out) *k_out = nullptr;
            if (v_out) *v_out = nullptr;
        } else {
            // Return cached values
            size_t cache_size = current_pos * num_heads * head_dim;
            k_temp.resize(cache_size);
            v_temp.resize(cache_size);
            
            for (int pos = 0; pos < current_pos; pos++) {
                size_t pos_offset = pos * num_heads * head_dim;
                size_t src_offset = base_offset + pos_offset;
                size_t dst_offset = pos * num_heads * head_dim;
                
                for (size_t h = 0; h < num_heads * head_dim; h++) {
                    k_temp[dst_offset + h] = k_cache[src_offset + h];
                    v_temp[dst_offset + h] = v_cache[src_offset + h + config.max_batch_size * config.max_seq_len * num_heads * head_dim];
                }
            }
            
            if (k_out) *k_out = k_temp.data();
            if (v_out) *v_out = v_temp.data();
        }
    }
    
    // Get cache size in bytes
    size_t size_bytes() const {
        return (k_cache.size() + v_cache.size()) * sizeof(float);
    }
    
    // Get memory usage in MB
    float size_mb() const {
        return size_bytes() / (1024.0f * 1024.0f);
    }
    
    // Check if cache is initialized
    bool is_initialized() const { return initialized; }
    
    // Get config
    const KVCacheConfig& get_config() const { return config; }
};

// Helper class for KV Cache with attention computation
class KVCacheAttention {
private:
    KVCache& cache;
    int layer;
    int batch_idx;
    int position;
    
public:
    KVCacheAttention(KVCache& c, int l, int b, int p) 
        : cache(c), layer(l), batch_idx(b), position(p) {}
    
    // Compute attention with cached K,V
    // This is a simplified version - full implementation would include
    // scaled dot-product attention
    void compute_attention(const float* q, const float* k_new, const float* v_new,
                          float* output, int num_heads, int head_dim) {
        // Store new K,V
        cache.store(layer, batch_idx, position, k_new, v_new);
        
        // Retrieve cached K,V
        float* k_cached = nullptr;
        float* v_cached = nullptr;
        int seq_len = 0;
        cache.retrieve(layer, batch_idx, position, &k_cached, &v_cached, &seq_len);
        
        // If first token, just use new K,V
        if (position == 0 || k_cached == nullptr) {
            // Self-attention with just current token
            // Simplified: output = V (since Q·K^T / sqrt(d) = 1 for same token)
            for (int i = 0; i < num_heads * head_dim; i++) {
                output[i] = v_new[i];
            }
            return;
        }
        
        // Compute attention scores: Q · K^T
        int total_dim = num_heads * head_dim;
        std::vector<float> scores(seq_len + 1, 0.0f);
        
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        // Score for cached tokens
        for (int i = 0; i <= seq_len; i++) {
            float score = 0.0f;
            const float* k_i = (i == seq_len) ? k_new : (k_cached + i * total_dim);
            
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    score += q[h * head_dim + d] * k_i[h * head_dim + d];
                }
            }
            scores[i] = score * scale;
        }
        
        // Softmax
        float max_score = scores[0];
        for (int i = 1; i <= seq_len; i++) {
            if (scores[i] > max_score) max_score = scores[i];
        }
        
        float sum = 0.0f;
        for (int i = 0; i <= seq_len; i++) {
            scores[i] = std::exp(scores[i] - max_score);
            sum += scores[i];
        }
        for (int i = 0; i <= seq_len; i++) {
            scores[i] /= sum;
        }
        
        // Weighted sum of V
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                float val = 0.0f;
                for (int i = 0; i <= seq_len; i++) {
                    const float* v_i = (i == seq_len) ? v_new : (v_cached + i * total_dim);
                    val += scores[i] * v_i[h * head_dim + d];
                }
                output[h * head_dim + d] = val;
            }
        }
    }
};

#endif // KV_CACHE_H
