/*
    ____  ___    ____  __________  ____  ___    ____  ____ 
   / __ \/   |  / __ \/_  __/ __ )/ __ \/   |  / __ \/ __ \
  / / / / /| | / /_/ / / / / __  / / / / /| | / /_/ / / / /
 / /_/ / ___ |/ _, _/ / / / /_/ / /_/ / ___ |/ _, _/ /_/ / 
/_____/_/  |_/_/ |_| /_/ /_____/\____/_/  |_/_/ |_/_____/  
                                                           
      A Cross-Platform Deep Learning Framework.

File: activationSigmoid.dart
Description: Sigmoid activation function. y=1/(1+e^{-x})

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
class ActivationSigmoid {
  //////////////////////// Member Vars

  //////////////////////////////// Member Funcs
  Matrix forward(Matrix inputs) {
    List<Iterable<double>> inp = inputs.toList();
    List<List<double>> toReturn = [];
    for (int i = 0; i < inp.length; i++) {
      List<double> working = [];
      List<double> inp2 = inp[i].toList();
      for (int j = 0; j < inp2.length; j++) {
        double output = 1.0 / (1 + exp(-inp2[j]));
        working.add(output);
      }
      toReturn.add(working);
    }
    return Matrix.fromList(toReturn);
  }

  Matrix backward(Matrix dvalues) {
    List<List<double>> toReturn = [];
    for (int i = 0; i < dvalues.length; i++) {
      List<double> working = [];
      List<double> inp2 = dvalues[i].toList();
      for (int j = 0; j < inp2.length; j++) {
        working.add(inp2[j] * (1.0 - inp2[j]));
      }
      toReturn.add(working);
    }
    return Matrix.fromList(toReturn);
  }
}
