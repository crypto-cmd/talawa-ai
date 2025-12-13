#!/bin/bash

echo "Cleaning..."

# Ensure we are in the project root
cd "$(dirname "$0")"

# Remove the directory
rm -rf build

# Recreate an empty one
mkdir build

echo "Build directory is now 100% empty."
