#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full end-to-end test of Neural LSH using the MNIST/SIFT binary datasets
and the LSH implementation from Project 1 (via /bin/search).
"""

import os
import subprocess
import time
from pathlib import Path
import sys

# Change paths if needed
DATASET_DIR = Path("../datasets/MNIST")   # or SIFT
BUILD_SCRIPT = "nlsh_build.py"
SEARCH_SCRIPT = "nlsh_search.py"
SEARCH_BIN = "../bin/search"              # Project 1 executable


def run_build():
    """
    Run nlsh_build.py using LSH-based k-NN graph.
    """
    print("\n Running Neural LSH Build...")

    cmd = [
        sys.executable, BUILD_SCRIPT,
        "-d", str(DATASET_DIR/"train-images.idx3-ubyte"),
        "-i", "test_index",
        "-type", "mnist",          # change to sift if testing SIFT
        "--knn", "5",
        "-m", "32",
        "--epochs", "3",
        "--layers", "2",
        "--nodes", "64",
        "--search_path", SEARCH_BIN
    ]

    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✔ Build phase completed.")


def run_search():
    """
    Run nlsh_search.py on actual query.dat dataset.
    """
    print("\n Running Neural LSH Search...")

    cmd = [
        sys.executable, SEARCH_SCRIPT,
        "-d", str(DATASET_DIR/"train-images.idx3-ubyte"),
        "-q", "../datasets/MNIST/t10k-images.idx3-ubyte",
        "-i", "test_index",
        "-o", "test_output.txt",
        "-type", "mnist",     # or "sift"
        "-N", "1",
        "-T", "3",
        "-range", "false"
    ]

    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✔ Search phase completed.")


def show_output():
    print("\n Contents of test_output.txt:\n")

    with open("test_output.txt", "r") as f:
        # Print only the first 80 lines to avoid spam
        for i, line in enumerate(f):
            print(line.rstrip())
            if i > 80:
                print("... (file truncated)")
                break


def main():
    start = time.time()

    print("===============================================")
    print("     Neural LSH – Full Pipeline Test (MNIST)   ")
    print("===============================================")

    run_build()
    run_search()
    show_output()

    print(f"\nTotal pipeline time: {time.time() - start:.2f} sec")


if __name__ == "__main__":
    main()
