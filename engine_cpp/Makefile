# Makefile for SmLM C++ Transformer Engine
# Supports: Linux, macOS, Termux/Android

# Detect platform
UNAME_S := $(shell uname -s)

# Compiler
CXX = g++

# Base flags
CXXFLAGS = -std=c++17 -O3 -Wall

# Platform-specific optimizations
ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -march=armv8-a+fp+simd
    LDFLAGS = -lm
else ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -march=armv8-a
    LDFLAGS = 
else
    LDFLAGS = -lm
endif

# Debug build
ifdef DEBUG
    CXXFLAGS = -std=c++17 -g -O0 -Wall -DDEBUG
endif

# Targets
TARGET = engine
TARGET_ENHANCED = enhanced_engine

.PHONY: all clean enhanced bpe test help

all: engine tokenizer bpe_tokenizer

engine: transformer_engine.cpp
	@echo "Building original engine..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: engine"

enhanced: enhanced_engine.cpp kv_cache.h quantize.h ngram.h graph_reasoning.h
	@echo "Building enhanced engine with all features..."
	$(CXX) $(CXXFLAGS) -o $(TARGET_ENHANCED) $< $(LDFLAGS)
	@echo "Build complete: enhanced_engine"

tokenizer: tokenizer.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

bpe_tokenizer: bpe_tokenizer.cpp
	@echo "Building BPE tokenizer..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: bpe_tokenizer"

test: engine tokenizer
	@echo "=== Running Tests ==="
	@echo "Engine: OK"
	@echo "Tokenizer: OK"
	@./engine --help

clean:
	rm -f engine enhanced_engine tokenizer bpe_tokenizer *.o

help:
	@echo "SmLM C++ Engine Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build everything (default)"
	@echo "  enhanced  - Build enhanced engine"
	@echo "  bpe       - Build BPE tokenizer"
	@echo "  clean     - Remove build artifacts"
	@echo ""
	@echo "  make DEBUG=1  - Debug build"
