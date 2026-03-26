// Graph Reasoning Layer for SmLM
// Adds graph-based reasoning on top of transformer representations
// Pure C++17 implementation

#ifndef GRAPH_REASONING_H
#define GRAPH_REASONING_H

#include <vector>
#include <map>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <string>

// Graph node representation
struct GraphNode {
    int id;
    std::vector<float> embedding;
    std::vector<int> neighbors;
    std::vector<float> edge_weights;
    std::string label;
    
    GraphNode() : id(0) {}
    GraphNode(int node_id, int dim) : id(node_id), embedding(dim, 0.0f) {}
};

// Graph edge
struct GraphEdge {
    int src;
    int dst;
    float weight;
    std::string relation;
    
    GraphEdge() : src(0), dst(0), weight(1.0f) {}
    GraphEdge(int s, int d) : src(s), dst(d), weight(1.0f) {}
    GraphEdge(int s, int d, float w) : src(s), dst(d), weight(w) {}
    GraphEdge(int s, int d, float w, const std::string& rel) 
        : src(s), dst(d), weight(w), relation(rel) {}
};

// Graph structure
class KnowledgeGraph {
private:
    std::map<int, GraphNode> nodes;
    std::vector<GraphEdge> edges;
    int embedding_dim = 0;
    
public:
    KnowledgeGraph(int dim = 128) : embedding_dim(dim) {}
    
    // Add node
    int add_node(const std::vector<float>& embedding = {}, const std::string& label = "") {
        int id = nodes.size();
        GraphNode node(id, embedding_dim);
        
        if (!embedding.empty()) {
            node.embedding = embedding;
        }
        node.label = label;
        
        nodes[id] = node;
        return id;
    }
    
    // Add edge
    void add_edge(int src, int dst, float weight = 1.0f, const std::string& relation = "") {
        if (nodes.find(src) == nodes.end() || nodes.find(dst) == nodes.end()) {
            return;
        }
        
        edges.push_back(GraphEdge(src, dst, weight, relation));
        nodes[src].neighbors.push_back(dst);
        nodes[src].edge_weights.push_back(weight);
    }
    
    // Get node
    GraphNode* get_node(int id) {
        auto it = nodes.find(id);
        if (it != nodes.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    // Get neighbors
    std::vector<int> get_neighbors(int id) const {
        auto it = nodes.find(id);
        if (it != nodes.end()) {
            return it->second.neighbors;
        }
        return {};
    }
    
    // Graph convolution step
    void graph_convolution(int iterations = 1, float learning_rate = 0.01f) {
        for (int iter = 0; iter < iterations; iter++) {
            std::map<int, std::vector<float>> new_embeddings;
            
            for (auto& [id, node] : nodes) {
                if (node.neighbors.empty()) continue;
                
                // Aggregate neighbor embeddings
                std::vector<float> aggregated(embedding_dim, 0.0f);
                float total_weight = 0.0f;
                
                for (size_t i = 0; i < node.neighbors.size(); i++) {
                    int neighbor_id = node.neighbors[i];
                    float weight = node.edge_weights[i];
                    
                    auto neighbor_it = nodes.find(neighbor_id);
                    if (neighbor_it != nodes.end()) {
                        for (int d = 0; d < embedding_dim; d++) {
                            aggregated[d] += neighbor_it->second.embedding[d] * weight;
                        }
                        total_weight += weight;
                    }
                }
                
                // Normalize
                if (total_weight > 0) {
                    for (int d = 0; d < embedding_dim; d++) {
                        aggregated[d] /= total_weight;
                    }
                }
                
                // Update embedding (residual connection)
                new_embeddings[id] = aggregated;
            }
            
            // Apply updates
            for (auto& [id, new_emb] : new_embeddings) {
                auto& node = nodes[id];
                for (int d = 0; d < embedding_dim; d++) {
                    node.embedding[d] = 0.8f * node.embedding[d] + 0.2f * new_emb[d];
                }
            }
        }
    }
    
    // Get node embedding
    std::vector<float> get_embedding(int id) const {
        auto it = nodes.find(id);
        if (it != nodes.end()) {
            return it->second.embedding;
        }
        return std::vector<float>(embedding_dim, 0.0f);
    }
    
    // Set node embedding
    void set_embedding(int id, const std::vector<float>& emb) {
        auto it = nodes.find(id);
        if (it != nodes.end() && static_cast<int>(emb.size()) == embedding_dim) {
            it->second.embedding = emb;
        }
    }
    
    // Number of nodes
    size_t num_nodes() const { return nodes.size(); }
    
    // Number of edges
    size_t num_edges() const { return edges.size(); }
    
    // Get all nodes
    const std::map<int, GraphNode>& get_nodes() const { return nodes; }
    
    // Get all edges
    const std::vector<GraphEdge>& get_edges() const { return edges; }
    
    // Save graph
    bool save(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file) return false;
        
        file.write("GRPH", 4);
        file.write(reinterpret_cast<const char*>(&embedding_dim), sizeof(embedding_dim));
        
        uint64_t num_nodes = nodes.size();
        file.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));
        
        for (const auto& [id, node] : nodes) {
            file.write(reinterpret_cast<const char*>(&id), sizeof(id));
            
            uint32_t label_size = node.label.size();
            file.write(reinterpret_cast<const char*>(&label_size), sizeof(label_size));
            file.write(node.label.data(), label_size);
            
            uint32_t emb_size = node.embedding.size();
            file.write(reinterpret_cast<const char*>(&emb_size), sizeof(emb_size));
            file.write(reinterpret_cast<const char*>(node.embedding.data()), 
                      emb_size * sizeof(float));
            
            uint32_t num_neighbors = node.neighbors.size();
            file.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));
            
            for (size_t i = 0; i < num_neighbors; i++) {
                file.write(reinterpret_cast<const char*>(&node.neighbors[i]), sizeof(int));
                file.write(reinterpret_cast<const char*>(&node.edge_weights[i]), sizeof(float));
            }
        }
        
        uint64_t num_edges = edges.size();
        file.write(reinterpret_cast<const char*>(&num_edges), sizeof(num_edges));
        
        for (const auto& edge : edges) {
            file.write(reinterpret_cast<const char*>(&edge.src), sizeof(int));
            file.write(reinterpret_cast<const char*>(&edge.dst), sizeof(int));
            file.write(reinterpret_cast<const char*>(&edge.weight), sizeof(float));
            
            uint32_t rel_size = edge.relation.size();
            file.write(reinterpret_cast<const char*>(&rel_size), sizeof(rel_size));
            file.write(edge.relation.data(), rel_size);
        }
        
        file.close();
        return true;
    }
    
    // Load graph
    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "GRPH") return false;
        
        file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(embedding_dim));
        
        uint64_t num_nodes;
        file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
        
        nodes.clear();
        edges.clear();
        
        for (uint64_t i = 0; i < num_nodes; i++) {
            int id;
            file.read(reinterpret_cast<char*>(&id), sizeof(id));
            
            uint32_t label_size;
            file.read(reinterpret_cast<char*>(&label_size), sizeof(label_size));
            std::string label(label_size, '\0');
            file.read(label.data(), label_size);
            
            uint32_t emb_size;
            file.read(reinterpret_cast<char*>(&emb_size), sizeof(emb_size));
            std::vector<float> emb(emb_size);
            file.read(reinterpret_cast<char*>(emb.data()), emb_size * sizeof(float));
            
            uint32_t num_neighbors;
            file.read(reinterpret_cast<char*>(&num_neighbors), sizeof(num_neighbors));
            
            GraphNode node(id, embedding_dim);
            node.label = label;
            node.embedding = emb;
            
            for (uint32_t j = 0; j < num_neighbors; j++) {
                int neighbor_id;
                float weight;
                file.read(reinterpret_cast<char*>(&neighbor_id), sizeof(int));
                file.read(reinterpret_cast<char*>(&weight), sizeof(float));
                node.neighbors.push_back(neighbor_id);
                node.edge_weights.push_back(weight);
            }
            
            nodes[id] = node;
        }
        
        uint64_t num_edges;
        file.read(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
        
        for (uint64_t i = 0; i < num_edges; i++) {
            GraphEdge edge;
            file.read(reinterpret_cast<char*>(&edge.src), sizeof(int));
            file.read(reinterpret_cast<char*>(&edge.dst), sizeof(int));
            file.read(reinterpret_cast<char*>(&edge.weight), sizeof(float));
            
            uint32_t rel_size;
            file.read(reinterpret_cast<char*>(&rel_size), sizeof(rel_size));
            edge.relation.resize(rel_size);
            file.read(edge.relation.data(), rel_size);
            
            edges.push_back(edge);
        }
        
        file.close();
        return true;
    }
};

// Graph-enhanced transformer
class GraphEnhancedTransformer {
private:
    KnowledgeGraph graph;
    int hidden_size;
    float graph_weight = 0.2f;  // Weight for graph contribution
    
public:
    GraphEnhancedTransformer(int hidden_dim, int graph_dim = 128) 
        : graph(graph_dim), hidden_size(hidden_dim) {}
    
    // Build graph from token embeddings
    void build_graph_from_tokens(const std::vector<int>& tokens,
                                 const std::vector<std::vector<float>>& token_embeddings) {
        // Add nodes for each token
        for (size_t i = 0; i < tokens.size(); i++) {
            // Project token embedding to graph embedding if dimensions differ
            std::vector<float> graph_emb(graph.num_nodes() > 0 ? 
                                         graph.get_node(0)->embedding.size() : hidden_size);
            
            if (i < token_embeddings.size()) {
                graph_emb = token_embeddings[i];
            }
            
            graph.add_node(graph_emb, "token_" + std::to_string(tokens[i]));
        }
        
        // Add edges between adjacent tokens
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            graph.add_edge(i, i + 1, 1.0f, "next");
        }
        
        // Add edges between same tokens (self-loops with higher weight)
        std::map<int, std::vector<int>> token_positions;
        for (size_t i = 0; i < tokens.size(); i++) {
            token_positions[tokens[i]].push_back(i);
        }
        
        for (const auto& [token, positions] : token_positions) {
            for (size_t i = 0; i < positions.size(); i++) {
                for (size_t j = i + 1; j < positions.size(); j++) {
                    // Add edge between same tokens
                    float weight = 0.5f / (j - i);  // Decay with distance
                    graph.add_edge(positions[i], positions[j], weight, "same");
                }
            }
        }
        
        // Run graph convolution
        graph.graph_convolution(2);
    }
    
    // Enhance transformer output with graph information
    std::vector<float> enhance_with_graph(const std::vector<float>& transformer_output,
                                          int token_position) {
        std::vector<float> enhanced = transformer_output;
        
        // Get graph embedding for this position
        std::vector<float> graph_emb = graph.get_embedding(token_position);
        
        // Blend transformer and graph embeddings
        if (graph_emb.size() == enhanced.size()) {
            for (size_t i = 0; i < enhanced.size(); i++) {
                enhanced[i] = (1.0f - graph_weight) * enhanced[i] + 
                              graph_weight * graph_emb[i];
            }
        }
        
        return enhanced;
    }
    
    // Get graph
    KnowledgeGraph& get_graph() { return graph; }
    
    // Set graph weight
    void set_graph_weight(float w) {
        graph_weight = std::max(0.0f, std::min(1.0f, w));
    }
};

#endif // GRAPH_REASONING_H
