#include "talawa-ai/neuralnetwork/DenseLayer.hpp"

#include <sstream>

namespace talawa_ai {
namespace nn {
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

Matrix DenseLayer::forward(const Matrix &input, bool is_training) {
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

Matrix DenseLayer::backward(const Matrix &outputGradients) {
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
std::vector<Matrix *> DenseLayer::getParameters() {
  return {&weights, &biases};
}

std::vector<Matrix *> DenseLayer::getParameterGradients() {
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

void DenseLayer::save(std::ostream &) const { return; }

void DenseLayer::load(std::istream &) { return; }

}  // namespace nn
}  // namespace talawa_ai
