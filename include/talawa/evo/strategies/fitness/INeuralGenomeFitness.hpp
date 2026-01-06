#pragma once
#include "talawa/evo/Genome.hpp"
#include "talawa/evo/NeuralGenome.hpp"
#include "talawa/evo/interfaces/IFitnessStrategy.hpp"
#include "talawa/evo/interfaces/IGenomeGeneratorStrategy.hpp"
#include "talawa/neuralnetwork/NeuralNetwork.hpp"

namespace talawa::evo {
using namespace talawa::nn;
class INeuralGenomeFitnessStrategy
    : public IFitnessStrategy<NeuralGenomeGeneType> {
 public:
  virtual ~INeuralGenomeFitnessStrategy() = default;

  // Calculate fitness for a NeuralGenome (must call setGenes to set weights
  // before use)
  virtual double calculateFitness(const Genome<NeuralGenomeGeneType>& ind) = 0;

 protected:
  void setGenes(NeuralGenome& genome, const NeuralGenomeGeneType& genes) {
    genome.setGenes(genes);  // Update genes vector
  }
};
}  // namespace talawa::evo
