#pragma once

#include "talawa/evo/Genome.hpp"
#include "talawa/evo/interfaces/IGenomeGeneratorStrategy.hpp"
#include "talawa/neuralnetwork/NeuralNetwork.hpp"
namespace talawa::evo {

using namespace talawa::nn;
using NeuralGenomeGeneType = std::vector<float>;
class NeuralGenome : public Genome<NeuralGenomeGeneType> {
 protected:
 private:
  NeuralNetwork brain;  // For running the network

 public:
  NeuralGenome& operator=(const NeuralGenome& other) {
    if (this != &other) {
      this->brain = other.brain;
      this->_genes = other._genes;
      this->fitness = other.fitness;
    }
    return *this;
  }
  NeuralGenome(const NeuralNetwork& nn) : brain(nn) {
    // Extract weights and biases from the neural network into genes
    _genes.clear();
    for (const auto& layer : brain.layers) {
      auto params = layer->getParameters();
      for (const auto& param : params) {
        for (size_t i = 0; i < param->rows; ++i) {
          for (size_t j = 0; j < param->cols; ++j) {
            _genes.push_back((*param)(i, j));
          }
        }
      }
    }
  }
  NeuralGenome(const NeuralGenome& other) : brain(other.brain) {
    this->_genes = other._genes;
    this->fitness = other.fitness;
  }

  Matrix predict(const Matrix& input) const { return brain.predict(input); }

  void setFitness(float fit) override { fitness = fit; }
  void setGenes(const NeuralGenomeGeneType& new_genes) {
    size_t gene_index = 0;
    for (auto& layer : brain.layers) {
      auto params = layer->getParameters();
      for (auto& param : params) {
        for (size_t i = 0; i < param->rows; ++i) {
          for (size_t j = 0; j < param->cols; ++j) {
            if (gene_index < new_genes.size()) {
              (*param)(i, j) = new_genes[gene_index++];
            } else {
              throw std::runtime_error(
                  "Not enough genes to set all parameters.");
            }
          }
        }
      }
    }
    if (gene_index != new_genes.size()) {
      throw std::runtime_error("Too many genes provided for the genome.");
    }
    // store the genes in the base Genome<T> member
    this->_genes = new_genes;
  }

  Genome<NeuralGenomeGeneType>& copy() const override {
    // copy neural network and flattened genes + fitness
    NeuralGenome* cloned = new NeuralGenome(*this);
    return *cloned;
  }
};
class NeuralGenomeGenerator
    : public IGenomeGeneratorStrategy<NeuralGenomeGeneType> {
  NeuralNetworkBuilder _topology;

 public:
  NeuralGenomeGenerator(const NeuralNetworkBuilder& topology)
      : _topology(topology) {}

  std::unique_ptr<Genome<NeuralGenomeGeneType>> generateGene() override {
    // Building consumes (moves) the builder's configs. Use a copy so the
    // original topology remains reusable for subsequent builds.
    auto builder_copy = _topology;  // uses copy ctor
    auto nn = builder_copy.build();
    return std::make_unique<NeuralGenome>(*nn);
  }
};
}  // namespace talawa::evo
