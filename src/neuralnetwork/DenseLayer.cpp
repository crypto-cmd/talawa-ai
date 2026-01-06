#include "talawa/neuralnetwork/DenseLayer.hpp"

#include <iostream>
#include <sstream>

namespace talawa {
namespace nn {
// Default constructor for load-time construction
DenseLayer::DenseLayer() : in(0), out(0) {}

DenseLayer::DenseLayer(size_t input_dim, size_t units, Activation act,
                       Initializer init)
    : in(input_dim), out(units) {
  // Set Hyperparameters
  this->activation = act;
  this->initializer = init;

  // 1. Initialize Weights (Input x Units)
  weights = Matrix(input_dim, units);
  initializer.apply(weights);

  // 2. Initialize Biases (1 x Units)
  biases = Matrix(1, units);
  Initializer(Initializer::ZEROS).apply(biases);

  // 3. Prepare gradient matrices with same shapes
  weights_grad = Matrix(input_dim, units);
  biases_grad = Matrix(1, units);
}

void DenseLayer::save(std::ostream& out) const {
  // Write dimensions
  out.write(reinterpret_cast<const char*>(&this->in), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(&this->out), sizeof(size_t));

  // Activation type
  int act = static_cast<int>(activation.type);
  out.write(reinterpret_cast<const char*>(&act), sizeof(int));

  // Save weights
  size_t w_rows = weights.rows;
  size_t w_cols = weights.cols;
  out.write(reinterpret_cast<const char*>(&w_rows), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(&w_cols), sizeof(size_t));
  size_t w_count = w_rows * w_cols;
  out.write(reinterpret_cast<const char*>(weights.rawData()),
            w_count * sizeof(float));

  // Save biases
  size_t b_rows = biases.rows;
  size_t b_cols = biases.cols;
  out.write(reinterpret_cast<const char*>(&b_rows), sizeof(size_t));
  out.write(reinterpret_cast<const char*>(&b_cols), sizeof(size_t));
  size_t b_count = b_rows * b_cols;
  out.write(reinterpret_cast<const char*>(biases.rawData()),
            b_count * sizeof(float));
}

void DenseLayer::load(std::istream& in_stream) {
  // Read dimensions
  size_t in_dim, out_dim;
  in_stream.read(reinterpret_cast<char*>(&in_dim), sizeof(size_t));
  in_stream.read(reinterpret_cast<char*>(&out_dim), sizeof(size_t));
  this->in = in_dim;
  this->out = out_dim;

  // Activation
  int act;
  in_stream.read(reinterpret_cast<char*>(&act), sizeof(int));
  this->activation = Activation(static_cast<Activation::Type>(act));

  // Read weights
  size_t w_rows, w_cols;
  in_stream.read(reinterpret_cast<char*>(&w_rows), sizeof(size_t));
  in_stream.read(reinterpret_cast<char*>(&w_cols), sizeof(size_t));
  this->weights = Matrix(static_cast<int>(w_rows), static_cast<int>(w_cols));
  size_t w_count = w_rows * w_cols;
  in_stream.read(reinterpret_cast<char*>(this->weights.rawData()),
                 w_count * sizeof(float));

  // Read biases
  size_t b_rows, b_cols;
  in_stream.read(reinterpret_cast<char*>(&b_rows), sizeof(size_t));
  in_stream.read(reinterpret_cast<char*>(&b_cols), sizeof(size_t));
  this->biases = Matrix(static_cast<int>(b_rows), static_cast<int>(b_cols));
  size_t b_count = b_rows * b_cols;
  in_stream.read(reinterpret_cast<char*>(this->biases.rawData()),
                 b_count * sizeof(float));

  // Recreate gradients and caches
  this->weights_grad =
      Matrix(static_cast<int>(in_dim), static_cast<int>(out_dim));
  this->biases_grad = Matrix(1, static_cast<int>(out_dim));
}

Matrix DenseLayer::forward(const Matrix& input, bool is_training) {
  if (is_training) {
    this->input_cache = input;
  }

  // Ensure Cache is Ready (Allocate ONCE, reuse forever)
  if (z_cache.rows != input.rows || z_cache.cols != out) {
    z_cache = Matrix(input.rows, out);
  }

  // Z - WX+b
  input.dot(weights, z_cache);
  z_cache += biases;

  // A - Activation(Z)
  auto a = activation.apply(z_cache);

  if (is_training) {
    this->a_cache = a;
  }

  return a;
}

Matrix DenseLayer::backward(const Matrix& outputGradients) {
  // 1. Calculate dL/dZ (Gradient through Activation)
  activation.backprop(a_cache, outputGradients, this->dZ);

  // dW = X^T * dZ
  input_cache.transpose(this->input_T_cache);
  input_T_cache.dot(this->dZ, this->weights_grad);

  // dB = sum(dZ, axis=0)
  biases_grad.fill(0.0f);
  this->dZ.sumRows(biases_grad);

  // dX = dZ * W^T
  weights.transpose(this->weights_T_cache);
  if (input_gradients_cache.rows != dZ.rows ||
      input_gradients_cache.cols != in) {
    input_gradients_cache = Matrix(dZ.rows, in);
  }
  this->dZ.dot(this->weights_T_cache, input_gradients_cache);

  return input_gradients_cache;
}

// --- Optimizers  ---
std::vector<Matrix*> DenseLayer::getParameters() { return {&weights, &biases}; }

std::vector<Matrix*> DenseLayer::getParameterGradients() {
  return {&weights_grad, &biases_grad};
}

// --- Utilities ---
Shape DenseLayer::getOutputShape() const {
  // Dense layers flatten dimensions: (1, 1, units)
  return {1, 1, out};
}

std::string DenseLayer::info() const {
  std::stringstream ss;
  ss << "Dense Layer [" << in << " -> " << out
     << "] Activation: " << activation.getName();
  return ss.str();
}

}  // namespace nn
}  // namespace talawa
