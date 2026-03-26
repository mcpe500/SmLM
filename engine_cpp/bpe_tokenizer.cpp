// BPE (Byte Pair Encoding) Tokenizer for SmLM
// Loads HuggingFace tokenizer.json and performs BPE tokenization
// Pure C++17 implementation

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <cstdint>
#include <regex>
#include <cmath>

class BPETokenizer {
private:
    std::map<std::string, int> vocab;
    std::map<int, std::string> id_to_token;
    std::vector<std::pair<std::string, std::string>> merges;
    std::map<std::string, int> merge_ranks;
    
    int vocab_size = 0;
    int unk_token_id = 0;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int pad_token_id = 3;
    
    std::string unk_token = "<|endoftext|>";
    std::string bos_token = "<|endoftext|>";
    std::string eos_token = "<|endoftext|>";
    std::string pad_token = "<|endoftext|>";
    
    // Regex patterns for pre-tokenization
    std::regex word_pattern;
    
public:
    BPETokenizer() : word_pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+)") {}
    
    // Load tokenizer from HuggingFace format
    bool load_from_file(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Error: Cannot open tokenizer file: " << path << std::endl;
            return false;
        }
        
        std::cout << "Loading tokenizer from: " << path << std::endl;
        
        // Simple JSON parsing (for tokenizer.json)
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        
        // Parse vocab
        parse_vocab(content);
        
        // Parse merges
        parse_merges(content);
        
        // Parse special tokens
        parse_special_tokens(content);
        
        std::cout << "Tokenizer loaded:" << std::endl;
        std::cout << "  - Vocab size: " << vocab_size << std::endl;
        std::cout << "  - Merges: " << merges.size() << std::endl;
        std::cout << "  - UNK token: " << unk_token_id << std::endl;
        std::cout << "  - BOS token: " << bos_token_id << std::endl;
        std::cout << "  - EOS token: " << eos_token_id << std::endl;
        
        return true;
    }
    
    void parse_vocab(const std::string& json) {
        // Find "vocab" section
        size_t vocab_start = json.find("\"vocab\"");
        if (vocab_start == std::string::npos) return;
        
        vocab_start = json.find("{", vocab_start);
        size_t vocab_end = json.find("}", vocab_start);
        
        if (vocab_start == std::string::npos || vocab_end == std::string::npos) return;
        
        std::string vocab_section = json.substr(vocab_start, vocab_end - vocab_start + 1);
        
        // Parse key-value pairs
        std::regex kv_pattern(R"(\"([^\"]+)\":\s*(\d+))");
        auto begin = std::sregex_iterator(vocab_section.begin(), vocab_section.end(), kv_pattern);
        auto end = std::sregex_iterator();
        
        for (auto it = begin; it != end; ++it) {
            std::string token = (*it)[1].str();
            int id = std::stoi((*it)[2].str());
            
            vocab[token] = id;
            id_to_token[id] = token;
            vocab_size = std::max(vocab_size, id + 1);
        }
    }
    
    void parse_merges(const std::string& json) {
        // Find "merges" section
        size_t merges_start = json.find("\"merges\"");
        if (merges_start == std::string::npos) return;
        
        merges_start = json.find("[", merges_start);
        size_t merges_end = json.find("]", merges_start);
        
        if (merges_start == std::string::npos || merges_end == std::string::npos) return;
        
        std::string merges_section = json.substr(merges_start, merges_end - merges_start + 1);
        
        // Parse merge pairs
        std::regex merge_pattern(R"(\"([^\"]+)\s+([^\"]+)\")");
        auto begin = std::sregex_iterator(merges_section.begin(), merges_section.end(), merge_pattern);
        auto end = std::sregex_iterator();
        
        int rank = 0;
        for (auto it = begin; it != end; ++it) {
            std::string first = (*it)[1].str();
            std::string second = (*it)[2].str();
            merges.push_back({first, second});
            merge_ranks[first + " " + second] = rank++;
        }
    }
    
    void parse_special_tokens(const std::string& json) {
        // Parse added_tokens
        std::regex token_pattern(R"(\"<\|endoftext\|>\":\s*(\d+))");
        std::smatch match;
        
        if (std::regex_search(json, match, token_pattern)) {
            int id = std::stoi(match[1].str());
            unk_token_id = id;
            bos_token_id = id;
            eos_token_id = id;
        }
        
        // Try to find pad_token if exists
        std::regex pad_pattern(R"(\"<\|padding\|>\":\s*(\d+))");
        if (std::regex_search(json, match, pad_pattern)) {
            pad_token_id = std::stoi(match[1].str());
        }
    }
    
    // Pre-tokenization: split text into words
    std::vector<std::string> pre_tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        
        // Simple whitespace and punctuation split
        std::string current;
        for (char c : text) {
            if (std::isspace(c) || std::ispunct(c)) {
                if (!current.empty()) {
                    tokens.push_back(current);
                    current.clear();
                }
                if (!std::isspace(c)) {
                    tokens.push_back(std::string(1, c));
                }
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            tokens.push_back(current);
        }
        
        return tokens;
    }
    
    // Get pairs from word
    std::vector<std::pair<std::string, int>> get_pairs(const std::vector<std::string>& word) {
        std::vector<std::pair<std::string, int>> pairs;
        for (size_t i = 0; i < word.size() - 1; i++) {
            pairs.push_back({word[i] + " " + word[i+1], (int)i});
        }
        return pairs;
    }
    
    // Apply BPE merges to a word
    std::string bpe(const std::string& word) {
        if (word.empty()) return "";
        
        // Split into characters
        std::vector<std::string> word_chars;
        for (char c : word) {
            word_chars.push_back(std::string(1, c));
        }
        
        // Apply merges iteratively
        while (word_chars.size() > 1) {
            // Find best merge
            int best_rank = INT32_MAX;
            int best_idx = -1;
            
            for (size_t i = 0; i < word_chars.size() - 1; i++) {
                std::string pair = word_chars[i] + " " + word_chars[i+1];
                auto it = merge_ranks.find(pair);
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                }
            }
            
            if (best_idx == -1) break;
            
            // Apply merge
            std::string merged = word_chars[best_idx] + word_chars[best_idx + 1];
            word_chars[best_idx] = merged;
            word_chars.erase(word_chars.begin() + best_idx + 1);
        }
        
        // Join tokens
        std::string result;
        for (const auto& c : word_chars) {
            if (!result.empty()) result += " ";
            result += c;
        }
        
        return result;
    }
    
    // Encode text to token IDs
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) {
        std::vector<int> tokens;
        
        // Pre-tokenize
        auto words = pre_tokenize(text);
        
        for (const auto& word : words) {
            // Apply BPE
            std::string bpe_result = bpe(word);
            
            // Convert to IDs
            std::istringstream iss(bpe_result);
            std::string token;
            while (iss >> token) {
                auto it = vocab.find(token);
                if (it != vocab.end()) {
                    tokens.push_back(it->second);
                } else {
                    // Try character-level fallback
                    bool found = false;
                    for (char c : token) {
                        std::string char_str(1, c);
                        auto char_it = vocab.find(char_str);
                        if (char_it != vocab.end()) {
                            tokens.push_back(char_it->second);
                            found = true;
                        }
                    }
                    if (!found) {
                        tokens.push_back(unk_token_id);
                    }
                }
            }
        }
        
        // Add special tokens
        if (add_special_tokens) {
            tokens.insert(tokens.begin(), bos_token_id);
            tokens.push_back(eos_token_id);
        }
        
        return tokens;
    }
    
    // Decode token IDs to text
    std::string decode(const std::vector<int>& tokens, bool skip_special_tokens = true) {
        std::ostringstream oss;
        
        for (size_t i = 0; i < tokens.size(); i++) {
            int id = tokens[i];
            
            // Skip special tokens if requested
            if (skip_special_tokens && 
                (id == bos_token_id || id == eos_token_id || id == pad_token_id)) {
                continue;
            }
            
            auto it = id_to_token.find(id);
            if (it != id_to_token.end()) {
                if (i > 0 && oss.tellp() > 0) oss << " ";
                oss << it->second;
            } else {
                oss << "<unk>";
            }
        }
        
        return oss.str();
    }
    
    // Get special token IDs
    int get_bos_id() const { return bos_token_id; }
    int get_eos_id() const { return eos_token_id; }
    int get_unk_id() const { return unk_token_id; }
    int get_pad_id() const { return pad_token_id; }
    int get_vocab_size() const { return vocab_size; }
    
    // Build vocabulary from text (for custom tokenizers)
    void build_vocab(const std::vector<std::string>& texts, int max_vocab_size = 50000) {
        std::map<std::string, int> char_freq;
        
        // Count character frequencies
        for (const auto& text : texts) {
            auto words = pre_tokenize(text);
            for (const auto& word : words) {
                for (char c : word) {
                    char_freq[std::string(1, c)]++;
                }
            }
        }
        
        // Assign IDs
        int id = 0;
        for (const auto& p : char_freq) {
            if (id >= max_vocab_size) break;
            vocab[p.first] = id;
            id_to_token[id] = p.first;
            id++;
        }
        
        vocab_size = id;
        std::cout << "Built vocabulary with " << vocab_size << " tokens" << std::endl;
    }
};

// CLI for BPE tokenizer
void print_usage(const char* program) {
    std::cout << "BPE Tokenizer for SmLM" << std::endl;
    std::cout << "Usage: " << program << " [command] [args]" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  encode <text>              - Encode text to token IDs" << std::endl;
    std::cout << "  decode <ids>               - Decode token IDs to text" << std::endl;
    std::cout << "  load <tokenizer.json>      - Load tokenizer from file" << std::endl;
    std::cout << "  test                       - Run basic tests" << std::endl;
    std::cout << "  help                       - Show this help" << std::endl;
}

int main(int argc, char* argv[]) {
    BPETokenizer tokenizer;
    
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string cmd = argv[1];
    
    if (cmd == "load" && argc > 2) {
        if (tokenizer.load_from_file(argv[2])) {
            std::cout << "Tokenizer loaded successfully" << std::endl;
        } else {
            std::cerr << "Failed to load tokenizer" << std::endl;
            return 1;
        }
        
    } else if (cmd == "encode" && argc > 2) {
        std::string text;
        for (int i = 2; i < argc; i++) {
            if (i > 2) text += " ";
            text += argv[i];
        }
        
        auto tokens = tokenizer.encode(text);
        std::cout << "Token IDs: ";
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << tokens[i];
        }
        std::cout << std::endl;
        std::cout << "Num tokens: " << tokens.size() << std::endl;
        
    } else if (cmd == "decode" && argc > 2) {
        std::string ids_str = argv[2];
        std::vector<int> ids;
        std::istringstream iss(ids_str);
        std::string token;
        
        while (std::getline(iss, token, ',')) {
            if (!token.empty()) {
                ids.push_back(std::stoi(token));
            }
        }
        
        std::string text = tokenizer.decode(ids);
        std::cout << "Text: " << text << std::endl;
        
    } else if (cmd == "test") {
        std::cout << "Running BPE tokenizer tests..." << std::endl;
        
        // Test basic encoding
        auto tokens = tokenizer.encode("hello world");
        std::cout << "Encode 'hello world': ";
        for (int t : tokens) std::cout << t << " ";
        std::cout << std::endl;
        
        // Test decoding
        std::string text = tokenizer.decode(tokens);
        std::cout << "Decode back: " << text << std::endl;
        
        std::cout << "Tests complete" << std::endl;
        
    } else if (cmd == "help" || cmd == "--help") {
        print_usage(argv[0]);
        
    } else {
        std::cerr << "Unknown command: " << cmd << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
