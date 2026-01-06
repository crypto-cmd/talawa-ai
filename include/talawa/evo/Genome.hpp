#pragma once

namespace talawa::evo {

template <typename T>
class Genome {
 protected:
  T _genes;

 public:
  double fitness = 0.0;
  ~Genome() = default;

  void setGenes(const T& genes) { _genes = genes; }
  const T& getGenes() const { return _genes; }
  float getFitness() const { return fitness; }
  virtual void setFitness(float fitness) { this->fitness = fitness; }

  virtual Genome<T>& copy() const {
    // Create a deep copy of the other genome
    Genome<T>* new_genome = new Genome<T>();
    new_genome->_genes = this->_genes;
    new_genome->fitness = this->fitness;
    return *new_genome;
  }
};
}  // namespace talawa::evo
