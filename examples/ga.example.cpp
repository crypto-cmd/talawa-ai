#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include "talawa/evo/Genome.hpp"
#include "talawa/evo/Population.hpp"
#include "talawa/evo/interfaces/ICrossoverStrategy.hpp"
#include "talawa/evo/interfaces/IFitnessStrategy.hpp"
#include "talawa/evo/interfaces/IGenomeGeneratorStrategy.hpp"
#include "talawa/evo/interfaces/IMutationSrategy.hpp"
#include "talawa/evo/interfaces/ISelectionStrategy.hpp"

using talawa::evo::Genome;
using talawa::evo::IGenomeGeneratorStrategy;
using talawa::evo::Population;
auto solution = "Maybe you will understand when you are older";

class GenomeGeneratorStrategy : public IGenomeGeneratorStrategy<std::string> {
 public:
  ~GenomeGeneratorStrategy() = default;
  std::unique_ptr<Genome<std::string>> generateGene() override {
    const char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789";
    const size_t max_index = (sizeof(charset) - 1);
    std::string gene;
    for (size_t i = 0; i < strlen(solution); i++) {
      gene += charset[rand() % max_index];
    }
    auto g = std::make_unique<Genome<std::string>>();
    g->setGenes(gene);
    return g;
  }
};

class FitnessStrategy : public talawa::evo::IFitnessStrategy<std::string> {
 public:
  ~FitnessStrategy() = default;
  double calculateFitness(
      const talawa::evo::Genome<std::string>& ind) override {
    const std::string& genes = ind.getGenes();
    double fitness = 0.0;
    for (size_t i = 0; i < genes.size(); i++) {
      if (genes[i] == solution[i]) {
        fitness += 1.0;
      } else {
        // Partial credit for being close
        fitness += 0.5 * (1.0 - (std::abs(genes[i] - solution[i]) / 128.0));
      }
    }
    return fitness;
  }
};

class CrossoverStrategy : public talawa::evo::ICrossoverStrategy<std::string> {
 public:
  ~CrossoverStrategy() = default;
  std::unique_ptr<talawa::evo::Genome<std::string>> crossover(
      const talawa::evo::Genome<std::string>& parent1,
      const talawa::evo::Genome<std::string>& parent2) override {
    // Handle empty parents
    if (parent1.getGenes().empty() || parent2.getGenes().empty()) {
      auto child = std::make_unique<talawa::evo::Genome<std::string>>();
      child->setGenes(parent1.getGenes().empty() ? parent2.getGenes()
                                                 : parent1.getGenes());
      return child;
    }
    // One-point crossover
    size_t point = rand() % parent1.getGenes().size();
    std::string child_genes =
        parent1.getGenes().substr(0, point) + parent2.getGenes().substr(point);
    // Create new Genome instance for child
    auto child = std::make_unique<talawa::evo::Genome<std::string>>();
    child->setGenes(child_genes);
    return child;
  }
};

class MutationStrategy : public talawa::evo::IMutationStrategy<std::string> {
 public:
  ~MutationStrategy() = default;
  void mutate(talawa::evo::Genome<std::string>& ind) override {
    auto genes = ind.getGenes();
    if (genes.empty()) return;
    size_t index = rand() % genes.size();
    const char charset[] =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789 ";
    const size_t max_index = (sizeof(charset) - 1);
    genes[index] = charset[rand() % max_index];
    ind.setGenes(genes);
  }
};

class SelectionStrategy : public talawa::evo::ISelectionStrategy<std::string> {
 public:
  ~SelectionStrategy() = default;
  const talawa::evo::Genome<std::string>& select(
      const std::vector<std::unique_ptr<talawa::evo::Genome<std::string>>>& pop)
      override {
    // Tournament selection
    size_t tournament_size = 5;
    const talawa::evo::Genome<std::string>* best = nullptr;
    for (size_t i = 0; i < tournament_size; i++) {
      const auto& contender_ptr = pop[rand() % pop.size()];
      if (!contender_ptr) continue;
      const talawa::evo::Genome<std::string>& contender = *contender_ptr;
      if (best == nullptr || contender.getFitness() > best->getFitness()) {
        best = &contender;
      }
    }
    return *best;
  }
};
int main() {
  std::srand(static_cast<unsigned>(std::time(nullptr)));
  // Inside main.cpp
  Population<std::string> pop(100);  // Create 100 random genomes

  // Further code to set strategies, initialize, and run the GA
  pop.setCrossoverStrategy(std::make_unique<CrossoverStrategy>());
  pop.setMutationStrategy(std::make_unique<MutationStrategy>());
  pop.setSelectionStrategy(std::make_unique<SelectionStrategy>());
  pop.setFitnessStrategy(std::make_unique<FitnessStrategy>());

  pop.initialize(std::make_unique<GenomeGeneratorStrategy>());

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
    printf("Generation %d: Best Fitness = %.2f | %s\n", generation,
           best_fitness, genomes[best_idx]->getGenes().c_str());
    if (best_fitness >= static_cast<double>(strlen(solution))) {
      printf("Solution found in generation %d: %s\n", generation,
             genomes[best_idx]->getGenes().c_str());
      break;
    }
  }

  return 0;
}
