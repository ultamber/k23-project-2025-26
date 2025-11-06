# ============================================================
# Makefile for Nearest Neighbor Search Project (LSH, Hypercube, IVFFlat, IVFPQ)
# ============================================================

CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Iinclude -Iinclude/algorithms -Iinclude/algorithms/clustering

SRC_DIR := src src/algorithms src/algorithms/clustering
INC_DIR := include
OBJ_DIR := obj
BIN_DIR := bin
RUN_DIR := runs

TARGET := $(BIN_DIR)/search

SRCS := $(wildcard $(addsuffix /*.cpp, $(SRC_DIR))) main.cpp
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: setup $(TARGET)

setup:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(RUN_DIR)

mnist:
	@$(TARGET) -d ./datasets/MNIST/input.dat \
	           -q ./datasets/MNIST/query.dat \
	           -o output.txt \
	           -type mnist -lsh -k 4 -L 10 -w 6 -N 1 -R 2000

sift:
	@$(TARGET) -d ./datasets/SIFT/input.dat \
	           -q ./datasets/SIFT/query.dat \
	           -o output.txt \
	           -type mnist -lsh -k 4 -L 10 -w 6 -N 1 -R 2

.PHONY: all setup clean run
