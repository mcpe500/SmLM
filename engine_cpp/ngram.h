// N-gram Language Model Support
// Provides n-gram based predictions that can be combined with transformer
// Pure C++17 implementation

#ifndef NGRAM_H
#define NGRAM_H

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Hash function for vector of ints (for n-gram key)
struct VectorHash {
    size_t operator()(const std::vector<int>& v) const {
        size_t seed = 0;
        for (int i : v) {
            seed ^= std::hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// N-gram model
class NGramModel {
private:
    int n;  // Order of n-gram
    std::unordered_map<std::vector<int>, std::map<int, int>, VectorHash> counts;
    std::unordered_map<std::vector<int>, int, VectorHash> context_counts;
    
    int vocab_size = 0;
    double smoothing_alpha = 1.0;  // Laplace smoothing
    
public:
    NGramModel(int order = 3) : n(order) {}
    
    // Train on token sequence
    void train(const std::vector<int>& tokens, int vocab_size) {
        this->vocab_size = vocab_size;
        
        // Count n-grams
        for (size_t i = 0; i < tokens.size(); i++) {
            // Get context (n-1 tokens)
            std::vector<int> context;
            for (int j = std::max(0, (int)i - n + 1); j < (int)i; j++) {
                context.push_back(tokens[j]);
            }
            
            // Pad context if needed
            while (context.size() < static_cast<size_t>(n - 1)) {
                context.insert(context.begin(), 0);  // PAD token
            }
            
            // Count
            counts[context][tokens[i]]++;
            context_counts[context]++;
        }
    }
    
    // Train on multiple sequences
    void train_batch(const std::vector<std::vector<int>>& sequences, int vocab_size) {
        this->vocab_size = vocab_size;
        
        for (const auto& seq : sequences) {
            train(seq, vocab_size);
        }
    }
    
    // Get probability of next token given context
    double probability(const std::vector<int>& context, int token_id) {
        // Get n-1 context
        std::vector<int> ctx;
        for (size_t i = std::max(0, (int)context.size() - n + 1); i < context.size(); i++) {
            ctx.push_back(context[i]);
        }
        
        // Pad if needed
        while (ctx.size() < static_cast<size_t>(n - 1)) {
            ctx.insert(ctx.begin(), 0);
        }
        
        // Get count
        auto ctx_it = counts.find(ctx);
        if (ctx_it == counts.end()) {
            // Unseen context - use uniform with smoothing
            return smoothing_alpha / (vocab_size + smoothing_alpha);
        }
        
        auto token_it = ctx_it->second.find(token_id);
        int count = (token_it != ctx_it->second.end()) ? token_it->second : 0;
        int total = context_counts[ctx];
        
        // Laplace smoothing
        return static_cast<double>(count + smoothing_alpha) / 
               static_cast<double>(total + smoothing_alpha * vocab_size);
    }
    
    // Get top-k most likely next tokens
    std::vector<std::pair<int, double>> topk(const std::vector<int>& context, int k) {
        // Get n-1 context
        std::vector<int> ctx;
        for (size_t i = std::max(0, (int)context.size() - n + 1); i < context.size(); i++) {
            ctx.push_back(context[i]);
        }
        
        while (ctx.size() < static_cast<size_t>(n - 1)) {
            ctx.insert(ctx.begin(), 0);
        }
        
        // Get counts for this context
        auto ctx_it = counts.find(ctx);
        if (ctx_it == counts.end()) {
            // Return uniform distribution
            std::vector<std::pair<int, double>> result;
            for (int i = 0; i < k && i < vocab_size; i++) {
                result.push_back({i, 1.0 / vocab_size});
            }
            return result;
        }
        
        // Calculate probabilities for all seen tokens
        std::vector<std::pair<int, double>> probs;
        int total = context_counts[ctx];
        
        for (const auto& p : ctx_it->second) {
            double prob = static_cast<double>(p.second + smoothing_alpha) / 
                         static_cast<double>(total + smoothing_alpha * vocab_size);
            probs.push_back({p.first, prob});
        }
        
        // Sort by probability
        std::sort(probs.begin(), probs.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Return top-k
        if (static_cast<int>(probs.size()) > k) {
            probs.resize(k);
        }
        
        return probs;
    }
    
    // Generate next token
    int predict(const std::vector<int>& context) {
        auto top = topk(context, 1);
        if (top.empty()) return 0;
        return top[0].first;
    }
    
    // Get model size (number of unique n-grams)
    size_t size() const {
        return counts.size();
    }
    
    // Get memory usage estimate
    size_t memory_bytes() const {
        size_t total = 0;
        for (const auto& p : counts) {
            total += p.first.size() * sizeof(int);
            total += p.second.size() * (sizeof(int) * 2);
        }
        return total;
    }
    
    // Set smoothing parameter
    void set_smoothing(double alpha) {
        smoothing_alpha = alpha;
    }
    
    // Save model to file
    bool save(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file) return false;
        
        // Write header
        file.write("NGRM", 4);  // N-gram model magic
        file.write(reinterpret_cast<const char*>(&n), sizeof(n));
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        
        // Write counts
        uint64_t num_contexts = counts.size();
        file.write(reinterpret_cast<const char*>(&num_contexts), sizeof(num_contexts));
        
        for (const auto& p : counts) {
            // Write context
            uint32_t ctx_size = p.first.size();
            file.write(reinterpret_cast<const char*>(&ctx_size), sizeof(ctx_size));
            file.write(reinterpret_cast<const char*>(p.first.data()), 
                      ctx_size * sizeof(int));
            
            // Write token counts
            uint32_t num_tokens = p.second.size();
            file.write(reinterpret_cast<const char*>(&num_tokens), sizeof(num_tokens));
            
            for (const auto& tp : p.second) {
                file.write(reinterpret_cast<const char*>(&tp.first), sizeof(int));
                file.write(reinterpret_cast<const char*>(&tp.second), sizeof(int));
            }
        }
        
        file.close();
        return true;
    }
    
    // Load model from file
    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        
        // Read header
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "NGRM") {
            return false;
        }
        
        file.read(reinterpret_cast<char*>(&n), sizeof(n));
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        
        uint64_t num_contexts;
        file.read(reinterpret_cast<char*>(&num_contexts), sizeof(num_contexts));
        
        counts.clear();
        context_counts.clear();
        
        for (uint64_t i = 0; i < num_contexts; i++) {
            uint32_t ctx_size;
            file.read(reinterpret_cast<char*>(&ctx_size), sizeof(ctx_size));
            
            std::vector<int> context(ctx_size);
            file.read(reinterpret_cast<char*>(context.data()), ctx_size * sizeof(int));
            
            uint32_t num_tokens;
            file.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
            
            for (uint32_t j = 0; j < num_tokens; j++) {
                int token_id, count;
                file.read(reinterpret_cast<char*>(&token_id), sizeof(int));
                file.read(reinterpret_cast<char*>(&count), sizeof(int));
                counts[context][token_id] = count;
            }
            
            // Recompute context count
            int total = 0;
            for (const auto& p : counts[context]) {
                total += p.second;
            }
            context_counts[context] = total;
        }
        
        file.close();
        return true;
    }
};

// Interpolation: combine n-gram with transformer logits
class NGramInterpolation {
private:
    NGramModel ngram;
    double ngram_weight = 0.3;  // Weight for n-gram (rest goes to transformer)
    
public:
    NGramInterpolation(int ngram_order = 3) : ngram(ngram_order) {}
    
    // Set n-gram weight
    void set_weight(double w) {
        ngram_weight = std::max(0.0, std::min(1.0, w));
    }
    
    // Train n-gram model
    void train(const std::vector<int>& tokens, int vocab_size) {
        ngram.train(tokens, vocab_size);
    }
    
    // Combine n-gram probabilities with transformer logits
    std::vector<float> combine(const std::vector<int>& context,
                               const float* transformer_logits,
                               int vocab_size) {
        std::vector<float> combined(vocab_size);
        
        // Get n-gram probabilities
        std::vector<double> ngram_probs(vocab_size, 0.0);
        for (int i = 0; i < vocab_size; i++) {
            ngram_probs[i] = ngram.probability(context, i);
        }
        
        // Normalize transformer logits (softmax)
        float max_logit = transformer_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (transformer_logits[i] > max_logit) max_logit = transformer_logits[i];
        }
        
        float sum = 0.0f;
        std::vector<float> transformer_probs(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            transformer_probs[i] = std::exp(transformer_logits[i] - max_logit);
            sum += transformer_probs[i];
        }
        for (int i = 0; i < vocab_size; i++) {
            transformer_probs[i] /= sum;
        }
        
        // Interpolate
        for (int i = 0; i < vocab_size; i++) {
            combined[i] = static_cast<float>(
                ngram_weight * ngram_probs[i] + 
                (1.0 - ngram_weight) * transformer_probs[i]
            );
        }
        
        return combined;
    }
    
    // Get n-gram model
    NGramModel& get_ngram() { return ngram; }
};

#endif // NGRAM_H
