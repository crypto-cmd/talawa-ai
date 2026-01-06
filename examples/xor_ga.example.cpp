#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "talawa/core/Matrix.hpp"
#include "talawa/evo/Genome.hpp"
#include "talawa/evo/NeuralGenome.hpp"  // Ensure NeuralGenome is a complete type here
#include "talawa/evo/Population.hpp"
#include "talawa/evo/interfaces/ICrossoverStrategy.hpp"
#include "talawa/evo/interfaces/IFitnessStrategy.hpp"
#include "talawa/evo/interfaces/IGenomeGeneratorStrategy.hpp"
#include "talawa/evo/interfaces/IMutationSrategy.hpp"
#include "talawa/evo/interfaces/ISelectionStrategy.hpp"
#include "talawa/evo/strategies/fitness/INeuralGenomeFitness.hpp"

using talawa::core::Activation;
using talawa::core::Matrix;
using talawa::evo::Genome;
using talawa::evo::IGenomeGeneratorStrategy;
using talawa::evo::NeuralGenome;
using talawa::evo::NeuralGenomeGenerator;
using talawa::evo::NeuralGenomeGeneType;
using talawa::evo::Population;
using talawa::nn::DenseLayerConfig;
using talawa::nn::NeuralNetworkBuilder;

auto solution = "Maybe you will understand when you are older";

class FitnessStrategy : public talawa::evo::INeuralGenomeFitnessStrategy {
 public:
  ~FitnessStrategy() = default;
  double calculateFitness(const Genome<NeuralGenomeGeneType>& ind) override {
    // SOlve XOR:
    auto inputs =
        Matrix({{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}});
    const auto expected = Matrix({{0.0f, 1.0f, 1.0f, 0.0f}});

    const NeuralGenome& neural_ind = dynamic_cast<const NeuralGenome&>(ind);
    double fitness = 0.0;
    Matrix outputs = neural_ind.predict(inputs);

    auto error = outputs - expected.transpose();
    float se = error.reduce<float>(
        [](float acc, int row, int col, float val) { return acc + val * val; },
        0.0f);
    fitness = 4.0f - se;

    return fitness;
  }
};

class CrossoverStrategy
    : public talawa::evo::ICrossoverStrategy<NeuralGenomeGeneType> {
 public:
  ~CrossoverStrategy() = default;
  std::unique_ptr<talawa::evo::Genome<NeuralGenomeGeneType>> crossover(
      const talawa::evo::Genome<NeuralGenomeGeneType>& parent1,
      const talawa::evo::Genome<NeuralGenomeGeneType>& parent2) override {
    // One-point crossover with safety guards
    auto p1_genes = parent1.getGenes();
    auto p2_genes = parent2.getGenes();

    // Choose crossover point using the smaller gene vector length
    size_t len = std::min(p1_genes.size(), p2_genes.size());
    size_t point = rand() % len;

    // Copy parent1 as NeuralGenome to create child
    auto child = std::make_unique<talawa::evo::NeuralGenome>(
        dynamic_cast<const NeuralGenome&>(parent1));

    // Swap random portion of genes from parent2
    NeuralGenomeGeneType child_genes = child->getGenes();
    for (size_t i = point; i < len; i++) {
      child_genes[i] = p2_genes[i];
    }
    child->setGenes(child_genes);
    return child;
  }
};

class MutationStrategy
    : public talawa::evo::IMutationStrategy<NeuralGenomeGeneType> {
 public:
  ~MutationStrategy() = default;
  void mutate(talawa::evo::Genome<NeuralGenomeGeneType>& ind) override {
    auto genes = ind.getGenes();
    if (genes.empty()) return;
    size_t index = rand() % genes.size();
    // Small random change
    float mutation_amount =
        ((static_cast<float>(rand()) / RAND_MAX) - 0.5f) * 0.2f;
    genes[index] += mutation_amount;
    ind.setGenes(genes);  // write mutated genes back into the genome
  }
};

class SelectionStrategy
    : public talawa::evo::ISelectionStrategy<NeuralGenomeGeneType> {
 public:
  ~SelectionStrategy() = default;
  const talawa::evo::Genome<NeuralGenomeGeneType>& select(
      const std::vector<
          std::unique_ptr<talawa::evo::Genome<NeuralGenomeGeneType>>>& pop)
      override {
    // Tournament selection
    size_t tournament_size = 5;
    const talawa::evo::Genome<NeuralGenomeGeneType>* best = nullptr;
    for (size_t i = 0; i < tournament_size; i++) {
      const auto& contender_ptr = pop[rand() % pop.size()];
      if (!contender_ptr) continue;
      const talawa::evo::Genome<NeuralGenomeGeneType>& contender =
          *contender_ptr;
      if (best == nullptr || contender.getFitness() > best->getFitness()) {
        best = &contender;
      }
    }
    return *best;
  }
};

static std::string genesToString(const NeuralGenomeGeneType& genes) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < genes.size(); ++i) {
    if (i) oss << ", ";
    oss << genes[i];
  }
  oss << "]";
  return oss.str();
}

int main() {
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  // Inside main.cpp
  Population<NeuralGenomeGeneType> pop(100);  // Create 100 random genomes

  // Further code to set strategies, initialize, and run the GA
  pop.setCrossoverStrategy(std::make_unique<CrossoverStrategy>());
  pop.setMutationStrategy(std::make_unique<MutationStrategy>());
  pop.setSelectionStrategy(std::make_unique<SelectionStrategy>());
  pop.setFitnessStrategy(std::make_unique<FitnessStrategy>());

  auto topology =
      NeuralNetworkBuilder::create({1, 1, 2})  // Input layer
          .add(DenseLayerConfig{10, Activation::TANH})
          .add(DenseLayerConfig{1, Activation::SIGMOID});  // Output layer

  pop.initialize(std::make_unique<NeuralGenomeGenerator>(topology));

  // Compute expected number of parameters once (building consumes the
  // builder's configs; use a copy to avoid emptying `topology`).
  //   auto builder_copy = topology;
  //   int total_parameters = builder_copy.build()->getTotalParameters();

  NeuralGenome best_genome_ever = dynamic_cast<const NeuralGenome&>(
      *pop.getGenomes()[0]);  // Initialize with first genome
  float best_fitness_ever = 0;
  for (int generation = 0; generation < 1000; generation++) {
    const auto& genomes = pop.step();

    // Find the best genome and its fitness
    double best_fitness = -1.0;
    size_t best_idx = 0;
    for (size_t i = 0; i < genomes.size(); i++) {
      double f = genomes[i]->getFitness();
      if (f > best_fitness) {
        best_fitness = f;
        best_idx = i;
      }
    }

    if (best_fitness > best_fitness_ever) {
      best_fitness_ever = static_cast<float>(best_fitness);
      best_genome_ever = dynamic_cast<NeuralGenome&>(*genomes[best_idx]);
    }

    std::cout << "Generation " << generation
              << ": Best Fitness = " << best_fitness << std::endl;
    if (best_fitness >= 3.9999f) {
      std::cout << "Solution found in generation " << generation << std::endl;
      break;
    }
  }

  // Test the best genome ever found so far
  auto test_inputs =
      Matrix({{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}});
  Matrix test_outputs = best_genome_ever.predict(test_inputs);
  std::cout << "Best Ever Genome Fitness: " << best_fitness_ever << std::endl;
  std::cout << "  Test Outputs of Best Ever Genome: ";
  test_inputs.print();
  test_outputs.print();

  return 0;
}
