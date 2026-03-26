// Simple tokenizer for SmLM C++ engine
// Provides basic token ID <-> text mapping for testing
// For production use, integrate with a real BPE tokenizer

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <sstream>

class SimpleTokenizer {
private:
    std::map<std::string, int> text_to_id;
    std::map<int, std::string> id_to_text;
    int vocab_size = 0;
    
public:
    SimpleTokenizer() {
        // Initialize with basic tokens for testing
        // In production, load from tokenizer.json or merges.txt
        
        // Special tokens
        add_token("<|endoftext|>", 50256);
        add_token("<|bos|>", 50257);
        add_token("<|eos|>", 50256);
        
        // Common English words (test vocabulary)
        std::vector<std::pair<std::string, int>> basic_tokens = {
            {"the", 100}, {"a", 101}, {"an", 102},
            {"is", 103}, {"are", 104}, {"was", 105}, {"were", 106},
            {"it", 107}, {"this", 108}, {"that", 109},
            {"I", 110}, {"you", 111}, {"he", 112}, {"she", 113},
            {"we", 114}, {"they", 115}, {"them", 116},
            {"have", 117}, {"has", 118}, {"had", 119},
            {"do", 120}, {"does", 121}, {"did", 122},
            {"be", 123}, {"been", 124}, {"being", 125},
            {"to", 126}, {"of", 127}, {"and", 128},
            {"in", 129}, {"on", 130}, {"at", 131},
            {"for", 132}, {"with", 133}, {"as", 134},
            {"from", 135}, {"by", 136}, {"or", 137},
            {"not", 138}, {"but", 139}, {"if", 140},
            {"can", 141}, {"will", 142}, {"would", 143},
            {"should", 144}, {"could", 145}, {"may", 146},
            {"what", 147}, {"when", 148}, {"where", 149},
            {"who", 150}, {"which", 151}, {"why", 152},
            {"how", 153}, {"all", 154}, {"each", 155},
            {"every", 156}, {"both", 157}, {"few", 158},
            {"more", 159}, {"most", 160}, {"other", 161},
            {"some", 162}, {"such", 163}, {"no", 164},
            {"only", 165}, {"own", 166}, {"same", 167},
            {"so", 168}, {"than", 169}, {"too", 170},
            {"very", 171}, {"just", 172}, {"also", 173},
            {"now", 174}, {"here", 175}, {"there", 176},
            {"then", 177}, {"once", 178}, {"again", 179},
            {"further", 180}, {"still", 181}, {"even", 182},
            {"back", 183}, {"out", 184}, {"up", 185},
            {"down", 186}, {"about", 187}, {"into", 188},
            {"over", 189}, {"after", 190}, {"before", 191},
            {"between", 192}, {"under", 193}, {"again", 194},
            {"then", 195}, {"once", 196}, {"here", 197},
            {"there", 198}, {"when", 199}, {"where", 200},
            {"why", 201}, {"how", 202}, {"all", 203},
            {"each", 204}, {"every", 205}, {"both", 206},
            {"few", 207}, {"more", 208}, {"most", 209},
            {"other", 210}, {"some", 211}, {"such", 212},
            {"no", 213}, {"nor", 214}, {"not", 215},
            {"only", 216}, {"own", 217}, {"same", 218},
            {"so", 219}, {"than", 220}, {"too", 221},
            {"very", 222}, {"can", 223}, {"will", 224},
            {"just", 225}, {"should", 226}, {"now", 227},
            // Numbers
            {"0", 300}, {"1", 301}, {"2", 302}, {"3", 303},
            {"4", 304}, {"5", 305}, {"6", 306}, {"7", 307},
            {"8", 308}, {"9", 309},
            // Common phrases
            {"hello", 400}, {"world", 401},
            {"model", 402}, {"language", 403},
            {"text", 404}, {"generation", 405},
            {"test", 406}, {"example", 407},
        };
        
        for (const auto& p : basic_tokens) {
            add_token(p.first, p.second);
        }
        
        vocab_size = 50258; // GPT-2 vocab size
    }
    
    void add_token(const std::string& text, int id) {
        text_to_id[text] = id;
        id_to_text[id] = text;
    }
    
    int encode(const std::string& text) {
        auto it = text_to_id.find(text);
        if (it != text_to_id.end()) {
            return it->second;
        }
        // Unknown token - return UNK or split into characters
        std::cerr << "Warning: Unknown token '" << text << "'" << std::endl;
        return 0; // UNK
    }
    
    std::vector<int> tokenize(const std::string& text) {
        std::vector<int> tokens;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // Remove punctuation
            word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
            if (!word.empty()) {
                tokens.push_back(encode(word));
            }
        }
        
        return tokens;
    }
    
    std::string decode(int token_id) {
        auto it = id_to_text.find(token_id);
        if (it != id_to_text.end()) {
            return it->second;
        }
        return "<unk>";
    }
    
    std::string detokenize(const std::vector<int>& tokens) {
        std::ostringstream oss;
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) oss << " ";
            oss << decode(tokens[i]);
        }
        return oss.str();
    }
    
    int get_vocab_size() const { return vocab_size; }
    
    // Load tokenizer from file (optional)
    bool load_from_file(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            std::cerr << "Warning: Cannot load tokenizer from " << path << std::endl;
            return false;
        }
        
        std::string line;
        int id = 0;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                add_token(line, id++);
            }
        }
        
        vocab_size = id;
        std::cout << "Loaded " << vocab_size << " tokens from " << path << std::endl;
        return true;
    }
};

// Standalone tokenizer CLI
void print_tokenizer_usage(const char* program) {
    std::cout << "Simple Tokenizer for SmLM" << std::endl;
    std::cout << "Usage: " << program << " [command] [args]" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  encode <text>     - Encode text to token IDs" << std::endl;
    std::cout << "  decode <ids>      - Decode token IDs to text (comma-separated)" << std::endl;
    std::cout << "  help              - Show this help" << std::endl;
}

int main(int argc, char* argv[]) {
    SimpleTokenizer tokenizer;
    
    if (argc < 2) {
        print_tokenizer_usage(argv[0]);
        return 1;
    }
    
    std::string cmd = argv[1];
    
    if (cmd == "encode" && argc > 2) {
        std::string text;
        for (int i = 2; i < argc; i++) {
            if (i > 2) text += " ";
            text += argv[i];
        }
        
        auto tokens = tokenizer.tokenize(text);
        std::cout << "Token IDs: ";
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << tokens[i];
        }
        std::cout << std::endl;
        
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
        
        std::string text = tokenizer.detokenize(ids);
        std::cout << "Text: " << text << std::endl;
        
    } else if (cmd == "help" || cmd == "--help") {
        print_tokenizer_usage(argv[0]);
        
    } else {
        std::cerr << "Unknown command: " << cmd << std::endl;
        print_tokenizer_usage(argv[0]);
        return 1;
    }
    
    return 0;
}
