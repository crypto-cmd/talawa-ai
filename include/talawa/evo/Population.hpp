#pragma once
#include <cstddef>
#include <memory>
#include <vector>

#include "talawa/evo/Genome.hpp"
#include "talawa/evo/interfaces/ICrossoverStrategy.hpp"
#include "talawa/evo/interfaces/IFitnessStrategy.hpp"
#include "talawa/evo/interfaces/IGenomeGeneratorStrategy.hpp"
#include "talawa/evo/interfaces/IMutationSrategy.hpp"
#include "talawa/evo/interfaces/ISelectionStrategy.hpp"

namespace talawa::evo {
template <typename T>
class Population {
 private:
  bool _initialized = false;

  // Buffers for double buffering
  std::vector<std::unique_ptr<Genome<T>>> _genomesA;
  std::vector<std::unique_ptr<Genome<T>>> _genomesB;

  size_t _size;

  std::unique_ptr<ISelectionStrategy<T>> selection;
  std::unique_ptr<ICrossoverStrategy<T>> crossover;
  std::unique_ptr<IMutationStrategy<T>> mutation;
  std::unique_ptr<IFitnessStrategy<T>> fitnessCalc;

 public:
  Population(size_t size) : _size(size) {
    _genomesA.resize(size);
    _genomesB.resize(size);
  }
  ~Population() = default;

  // Accessors for the active and next-generation buffers
  std::vector<std::unique_ptr<Genome<T>>>& getGenomes() { return _genomesA; }
  const std::vector<std::unique_ptr<Genome<T>>>& getGenomes() const {
    return _genomesA;
  }
  std::vector<std::unique_ptr<Genome<T>>>& getNewGenomes() { return _genomesB; }

  // Methods to inject strategies (Dependency Injection)
  void setSelectionStrategy(std::unique_ptr<ISelectionStrategy<T>> s) {
    selection = std::move(s);
  }
  void setCrossoverStrategy(std::unique_ptr<ICrossoverStrategy<T>> c) {
    crossover = std::move(c);
  }
  void setMutationStrategy(std::unique_ptr<IMutationStrategy<T>> m) {
    mutation = std::move(m);
  }
  void setFitnessStrategy(std::unique_ptr<IFitnessStrategy<T>> f) {
    fitnessCalc = std::move(f);
  }
  void initialize(std::unique_ptr<IGenomeGeneratorStrategy<T>> generator) {
    for (size_t i = 0; i < getGenomes().size(); i++) {
      getGenomes()[i] = generator->generateGene();
    }
    _initialized = true;
  }

  std::vector<std::unique_ptr<Genome<T>>>& step() {
    if (!_initialized) {
      throw std::runtime_error("Population not initialized with genes.");
    }
    if (!selection || !crossover || !mutation || !fitnessCalc) {
      throw std::runtime_error("Population strategies not fully configured.");
    }

    // Evaluate fitness of the current (active) generation - needed by selection
    evaluateFitness();

    // Create new generation in the next buffer
    createNewGeneration();

    // Swap buffers so getGenomes() returns the new generation
    std::swap(_genomesA, _genomesB);

    // Fitness for the new generation is already calculated during creation
    return getGenomes();
  }

 private:
  void evaluateFitness() {
    for (auto& ind_ptr : getGenomes()) {
      if (!ind_ptr) continue;
      double fitness = fitnessCalc->calculateFitness(*ind_ptr);
      ind_ptr->setFitness(static_cast<float>(fitness));
      
    }
  }

  void createNewGeneration() {
    for (size_t i = 0; i < _size; i++) {
      const Genome<T>& parent1 = selection->select(getGenomes());
      const Genome<T>& parent2 = selection->select(getGenomes());
      auto offspring = crossover->crossover(parent1, parent2);
      if (!offspring) {
        throw std::runtime_error("Crossover returned null offspring.");
      }
      mutation->mutate(*offspring);
      // Calculate fitness for the offspring now so we don't need a global
      // re-evaluate
      double offspringFitness = fitnessCalc->calculateFitness(*offspring);
      offspring->setFitness(static_cast<float>(offspringFitness));
      getNewGenomes()[i] = std::move(offspring);
    }
  }
};
}  // namespace talawa::evo
