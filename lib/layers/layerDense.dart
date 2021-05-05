/*
    ____  ___    ____  __________  ____  ___    ____  ____ 
   / __ \/   |  / __ \/_  __/ __ )/ __ \/   |  / __ \/ __ \
  / / / / /| | / /_/ / / / / __  / / / / /| | / /_/ / / / /
 / /_/ / ___ |/ _, _/ / / / /_/ / /_/ / ___ |/ _, _/ /_/ / 
/_____/_/  |_/_/ |_| /_/ /_____/\____/_/  |_/_/ |_/_____/  
                                                           
      A Cross-Platform Deep Learning Framework.

File: layerDense.dart
Description: Foundational Element that makes up a multi-layer perceptron Layer.

“Commons Clause” License Condition v1.0
The Software is provided to you by the Licensor under the License, 
as defined below, subject to the following condition.
Without limiting other conditions in the License, the grant of rights under
the License will not include, and the License does not grant to you, the right
to Sell the Software.

For purposes of the foregoing, “Sell” means practicing any or all of the rights
granted to you under the License to provide to third parties, for a fee or other
consideration (including without limitation fees for hosting or consulting/ 
support services related to the Software), a product or service whose value 
derives, entirely or substantially, from the functionality of the Software.
Any license notice or attribution required by the License must also include 
this Commons Clause License Condition notice.

Software: DartBoard
License: MIT
Licensor: Seth Kitchen 2021
WHAT DOES THIS MEAN?

You must pay me to use this in a commerical setting -- 
contact me seth [at] collaboarator.com
*/
library dartboard;

////////////////////////////////////////////////////// Imports
import 'dart:math';
import 'package:ml_linalg/matrix.dart';

///////////////////////////////////////////////////// Class Decl
class LayerDense {
  //////////////////////// Member Vars
  Random r = Random();
  Matrix? weights;
  Matrix? dweights;
  Matrix? biases;
  Matrix? dbiases;
  Matrix? _inputs;
  Matrix? dinputs;
  int? batchSize;

  //////////////////////// Constructors
  LayerDense(int numInputs, int numNeurons, {required this.batchSize}) {
    List<List<double>> matrixBuilder = [];
    List<List<double>> biasBuilder = [];
    for (int i = 0; i < numInputs; i++) {
      List<double> inputToNeuron = [];
      for (int j = 0; j < numNeurons; j++) {
        inputToNeuron.add(r.nextDouble());
      }
      matrixBuilder.add(inputToNeuron);
    }
    for (int i = 0; i < batchSize!; i++) {
      List<double> bias = [];
      for (int j = 0; j < numNeurons; j++) {
        bias.add(0);
      }
      biasBuilder.add(bias);
    }
    weights = Matrix.fromList(matrixBuilder);
    biases = Matrix.fromList(biasBuilder);
  }

  LayerDense.withInitNeuron(
      int numInputs, int numNeurons, double initNeuronValue,
      {required this.batchSize}) {
    List<List<double>> matrixBuilder = [];
    List<List<double>> biasBuilder = [];
    for (int i = 0; i < numInputs; i++) {
      List<double> inputToNeuron = [];
      for (int j = 0; j < numNeurons; j++) {
        inputToNeuron.add(initNeuronValue);
      }
      matrixBuilder.add(inputToNeuron);
    }
    for (int i = 0; i < batchSize!; i++) {
      List<double> bias = [];
      for (int j = 0; j < numNeurons; j++) {
        bias.add(0);
      }
      biasBuilder.add(bias);
    }
    weights = Matrix.fromList(matrixBuilder);
    biases = Matrix.fromList(biasBuilder);
  }

  LayerDense.testMatrixCalcs({required this.biases, required this.weights}) {
    batchSize = this.biases!.rowsNum;
  }

  //////////////////////////////// Member Funcs
  Matrix forward(Matrix inputs) {
    _inputs = inputs;
    Matrix toReturn = inputs * weights! + biases!;
    return toReturn;
  }

  List<Matrix> backward(Matrix dvalues) {
    List<Matrix> toReturn = [];
    assert(_inputs != null);
    assert(batchSize != null);
    /////////////////////////////////////////// Weights Derivative
    dweights = _inputs!.transpose() * dvalues;
    toReturn.add(dweights!);

    /////////////////////////////////////////// Bias Derivative
    List<List<double>> workingAllBiases = [];
    List<double> workingBias = [];
    dvalues.columns.forEach((element) {
      workingBias.add(element.sum());
    });
    for (int i = 0; i < batchSize!; i++) {
      workingAllBiases.add(workingBias);
    }
    toReturn.add(Matrix.fromList(workingAllBiases));

    ////////////////////////////////////////// Inputs Derivative
    dinputs = dvalues * weights!.transpose();
    toReturn.add(dinputs!);

    return toReturn;
  }
}
