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
    # Linux (including Termux/Android)
    CXXFLAGS += -march=armv8-a+fp+simd
    LDFLAGS = -lm
else ifeq ($(UNAME_S),Darwin)
    # macOS
    CXXFLAGS += -march=armv8-a
    LDFLAGS = 
else
    # Generic
    LDFLAGS = -lm
endif

# Debug build (optional)
ifdef DEBUG
    CXXFLAGS = -std=c++17 -g -O0 -Wall -DDEBUG
endif

# Targets
TARGET = engine
TARGET_FULL = transformer_engine
SRC = transformer_engine.cpp

# Test data generator
TEST_GEN = test_tokenizer
TEST_GEN_SRC = test_tokenizer.cpp

.PHONY: all clean install debug test

all: $(TARGET)

# Main engine
$(TARGET): $(SRC)
	@echo "Building $(TARGET)..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"
	@echo "Run with: ./$(TARGET) --help"

# Debug build
debug: CXXFLAGS = -std=c++17 -g -O0 -Wall -DDEBUG
debug: $(TARGET)

# Tokenizer utility
TOKENIZER_SRC = tokenizer.cpp
TOKENIZER = tokenizer

$(TOKENIZER): $(TOKENIZER_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

test: $(TARGET) $(TOKENIZER)
	@echo "Running tests..."
	@echo "Engine build: OK"
	@echo "Tokenizer build: OK"
	@./$(TARGET) --help
	@echo ""
	@./$(TOKENIZER) help

clean:
	rm -f $(TARGET) $(TOKENIZER) *.o

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
	@echo "Installed to /usr/local/bin/$(TARGET)"

# Help
help:
	@echo "SmLM C++ Engine Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build engine (default)"
	@echo "  debug   - Build with debug symbols"
	@echo "  test    - Build and run basic tests"
	@echo "  clean   - Remove build artifacts"
	@echo "  install - Install to /usr/local/bin"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build for current platform"
	@echo "  make DEBUG=1  - Build debug version"
	@echo "  ./engine --model model.slm --input 50256,123,456"
