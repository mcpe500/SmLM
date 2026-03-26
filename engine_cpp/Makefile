CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall
LDFLAGS = 

# Optional: ONNX Runtime
# ONNXRUNTIME_DIR = /usr/local/onnxruntime
# CXXFLAGS += -I$(ONNXRUNTIME_DIR)/include
# LDFLAGS += -L$(ONNXRUNTIME_DIR)/lib -lonnxruntime

TARGET = engine
SRC = engine.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/
