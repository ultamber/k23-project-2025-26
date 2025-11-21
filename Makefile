# ============================================================
# Makefile for Nearest Neighbor Search Project (LSH, Hypercube, IVFFlat, IVFPQ)
# ============================================================

CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Iinclude

SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
BIN_DIR := bin
RUN_DIR := runs

TARGET := $(BIN_DIR)/search
SRC_FILES := $(shell find $(SRC_DIR) -name '*.cpp') main.cpp
OBJ_FILES := $(SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJ_FILES) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(RUN_DIR)/**/*.txt

mnist:
	@$(TARGET) -d ./datasets/MNIST/input.dat \
	           -q ./datasets/MNIST/query.dat \
	           -o output.txt \
		   -gt ./datasets/MNIST/ground_truth.csv \
	           -type mnist -lsh -k 4 -L 10 -w 6 -N 1 -R 2000

sift:
	@$(TARGET) -d ./datasets/SIFT/input.dat \
	           -q ./datasets/SIFT/query.dat \
	           -o output.txt \
		   -gt ./datasets/SIFT/ground_truth.csv \
	           -type mnist -lsh -k 4 -L 10 -w 6 -N 1 -R 2

.PHONY: all setup clean run
