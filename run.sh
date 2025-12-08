#!/bin/bash
set -e

echo "--- Building Project ---"
# 1. Build
mkdir -p build && cd build
cmake .. > /dev/null
cmake --build .

# Copy all files from data directory to build directory
if [ -d "../data" ]; then
    cp -r ../data/* .
fi
# 2. Run
# If the user typed a name (e.g., ./run.sh matrix.test), run that.
if [ -n "$1" ]; then
    echo "--- Running Target: $1 ---"
    ./$1
else
    # Default: Try to run 'main' if no argument is given
    if [ -f "./main" ]; then
        echo "--- Running Main ---"
        ./main
    else
        echo "Build success! (No executable specified to run)"
    fi
fi
