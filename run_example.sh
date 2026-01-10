#!/usr/bin/env bash
# Usage: ./run_example.sh examples/main.example
# Builds (if needed) and runs the specified example

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <example_source_file>"
    exit 1
fi


EXAMPLE_SRC="$1"
EXAMPLE_NAME=$(basename "$EXAMPLE_SRC")
EXAMPLE_TARGET="$EXAMPLE_NAME"
EXAMPLE_BIN="build/$EXAMPLE_NAME"

# Check if build directory exists
if [ ! -d build ]; then
    echo "[INFO] Build directory not found. Creating and configuring with CMake..."
    mkdir -p build
fi


cd build
cmake ..
cd ..


# Build only the requested example target
echo "[INFO] Building target $EXAMPLE_TARGET..."
cmake --build build --target "$EXAMPLE_TARGET" -- -j

# Check if the binary exists
if [ ! -x "$EXAMPLE_BIN" ]; then
    echo "[ERROR] Example binary '$EXAMPLE_BIN' not found after build."
    exit 2
fi

echo "[INFO] Running $EXAMPLE_BIN..."
"$EXAMPLE_BIN"
