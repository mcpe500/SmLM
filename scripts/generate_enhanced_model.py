#!/usr/bin/env python3
"""Generate Enhanced SmLM Model with ALL features"""

import struct
import json
import random
import math

random.seed(42)

# Config
VOCAB_SIZE = 5000
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
INTERMEDIATE_SIZE = 1024
MAX_POSITIONS = 512

def random_tensor(size, scale=0.02):
    return [random.gauss(0, scale) for _ in range(size)]

def write_tensor(f, data):
    f.write(struct.pack('<I', len(data)))
    if data:
        f.write(struct.pack(f'<{len(data)}f', *data))

print("=== Generating Enhanced SmLM Model ===")
print(f"Config: {NUM_LAYERS}L/{HIDDEN_SIZE}H/{NUM_HEADS}A")

# Build tokenizer
print("\n1. Building BPE tokenizer...")
vocab = {}
for i in range(256):
    vocab[f"chr{i}"] = i
for i in range(256, VOCAB_SIZE):
    vocab[f"tok{i}"] = i

tokenizer_json = {
    "version": "1.0",
    "model": {"type": "BPE"},
    "vocab": vocab,
    "merges": []
}

with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer_json, f)
print(f"   Saved tokenizer.json ({len(vocab)} tokens)")

# Build model
print("\n2. Building transformer weights...")
with open('enhanced_model.slm', 'wb') as f:
    # Magic: ESLM (Enhanced SmLM)
    f.write(b'ESLM')
    f.write(struct.pack('<I', 1))  # version
    
    # Config
    f.write(struct.pack('<I', VOCAB_SIZE))
    f.write(struct.pack('<I', MAX_POSITIONS))
    f.write(struct.pack('<I', HIDDEN_SIZE))
    f.write(struct.pack('<I', NUM_LAYERS))
    f.write(struct.pack('<I', NUM_HEADS))
    f.write(struct.pack('<I', INTERMEDIATE_SIZE))
    
    # Feature flags
    features = 0b1111  # All features enabled
    f.write(struct.pack('<I', features))
    
    # Embeddings
    print("   Writing embeddings...")
    write_tensor(f, random_tensor(VOCAB_SIZE * HIDDEN_SIZE))
    write_tensor(f, random_tensor(MAX_POSITIONS * HIDDEN_SIZE))
    
    # Layers
    print(f"   Writing {NUM_LAYERS} transformer layers...")
    for layer in range(NUM_LAYERS):
        # QKV
        write_tensor(f, random_tensor(HIDDEN_SIZE * HIDDEN_SIZE))  # Q
        write_tensor(f, [0.0] * HIDDEN_SIZE)
        write_tensor(f, random_tensor(HIDDEN_SIZE * HIDDEN_SIZE))  # K
        write_tensor(f, [0.0] * HIDDEN_SIZE)
        write_tensor(f, random_tensor(HIDDEN_SIZE * HIDDEN_SIZE))  # V
        write_tensor(f, [0.0] * HIDDEN_SIZE)
        # Output
        write_tensor(f, random_tensor(HIDDEN_SIZE * HIDDEN_SIZE))
        write_tensor(f, [0.0] * HIDDEN_SIZE)
        # Layer norms
        write_tensor(f, [1.0] * HIDDEN_SIZE)
        write_tensor(f, [0.0] * HIDDEN_SIZE)
        write_tensor(f, [1.0] * HIDDEN_SIZE)
        write_tensor(f, [0.0] * HIDDEN_SIZE)
        # MLP
        write_tensor(f, random_tensor(HIDDEN_SIZE * INTERMEDIATE_SIZE))
        write_tensor(f, [0.0] * INTERMEDIATE_SIZE)
        write_tensor(f, random_tensor(INTERMEDIATE_SIZE * HIDDEN_SIZE))
        write_tensor(f, [0.0] * HIDDEN_SIZE)
    
    # Final norm
    print("   Writing final norm and LM head...")
    write_tensor(f, [1.0] * HIDDEN_SIZE)
    write_tensor(f, [0.0] * HIDDEN_SIZE)
    write_tensor(f, random_tensor(HIDDEN_SIZE * VOCAB_SIZE))
    write_tensor(f, [0.0] * VOCAB_SIZE)

print("   Saved enhanced_model.slm")

# Build N-gram model
print("\n3. Building N-gram model...")
with open('model.ngram', 'wb') as f:
    f.write(b'NGRM')
    f.write(struct.pack('<I', 3))  # order
    f.write(struct.pack('<I', VOCAB_SIZE))
    
    # Write some sample n-grams
    ngrams = {}
    for i in range(1000):
        ctx = tuple(random.randint(0, 100) for _ in range(3))
        token = random.randint(0, VOCAB_SIZE-1)
        ngrams[ctx] = ngrams.get(ctx, 0) + 1
    
    f.write(struct.pack('<Q', len(ngrams)))
    for ctx, count in ngrams.items():
        f.write(struct.pack('<I', len(ctx)))
        for t in ctx:
            f.write(struct.pack('<I', t))
        f.write(struct.pack('<I', count))

print("   Saved model.ngram")

# Build Graph
print("\n4. Building knowledge graph...")
with open('model.graph', 'wb') as f:
    f.write(b'GRPH')
    f.write(struct.pack('<I', 128))  # dim
    
    # Write nodes
    num_nodes = 100
    f.write(struct.pack('<Q', num_nodes))
    for i in range(num_nodes):
        f.write(struct.pack('<I', i))
        label = f"node{i}"
        f.write(struct.pack('<I', len(label)))
        f.write(label.encode())
        emb = random_tensor(128)
        f.write(struct.pack('<I', len(emb)))
        f.write(struct.pack(f'<{len(emb)}f', *emb))
        f.write(struct.pack('<I', 0))  # neighbors
    
    # Write edges
    num_edges = 200
    f.write(struct.pack('<Q', num_edges))
    for _ in range(num_edges):
        src = random.randint(0, num_nodes-1)
        dst = random.randint(0, num_nodes-1)
        f.write(struct.pack('<i', src))
        f.write(struct.pack('<i', dst))
        f.write(struct.pack('<f', 1.0))
        rel = "next"
        f.write(struct.pack('<I', len(rel)))
        f.write(rel.encode())

print("   Saved model.graph")

# Summary
print("\n=== Summary ===")
print("Files created:")
print("  - enhanced_model.slm  (main model)")
print("  - tokenizer.json      (BPE tokenizer)")
print("  - model.ngram         (N-gram model)")
print("  - model.graph         (Knowledge graph)")
print("\nUsage:")
print("  ./enhanced_engine \\")
print("    --model enhanced_model.slm \\")
print("    --tokenizer tokenizer.json \\")
print("    --ngram-model model.ngram \\")
print("    --graph-model model.graph \\")
print("    --input 0,1,2 \\")
print("    --kv-cache --ngram --graph")
