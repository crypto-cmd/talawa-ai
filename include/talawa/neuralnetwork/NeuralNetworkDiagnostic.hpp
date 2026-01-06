#pragma once

#include <string>
#include <vector>

#include "talawa/neuralnetwork/Loss.hpp"
#include "talawa/neuralnetwork/NeuralNetwork.hpp"

namespace talawa {
namespace nn {

// Severity levels for NeuralNetworkDiagnostics
enum class IssueSeverity {
  INFO,
  WARNING,
  CRITICAL  // Will likely cause NaN/Crash
};

struct NeuralNetworkDiagnosticIssue {
  IssueSeverity severity;
  std::string title;
  std::string message;
  std::string suggestion;
};

class NeuralNetworkDiagnostic {
 public:
  /**
   * Run all checks on the network and loss function.
   * Prints issues directly to std::cerr.
   * returns: true if CRITICAL issues were found.
   */
  static bool check(const nn::NeuralNetwork& net, const nn::loss::Loss& loss);

 private:
  // Individual Checks
  static std::vector<NeuralNetworkDiagnosticIssue> checkStructure(
      const nn::NeuralNetwork& net);
  static std::vector<NeuralNetworkDiagnosticIssue> checkCompatibility(
      const nn::NeuralNetwork& net, const nn::loss::Loss& loss);

  // Helper to format output
  static void printReport(
      const std::vector<NeuralNetworkDiagnosticIssue>& issues);
};

}  // namespace nn
}  // namespace talawa
