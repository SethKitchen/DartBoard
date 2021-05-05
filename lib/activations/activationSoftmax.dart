/*
    ____  ___    ____  __________  ____  ___    ____  ____ 
   / __ \/   |  / __ \/_  __/ __ )/ __ \/   |  / __ \/ __ \
  / / / / /| | / /_/ / / / / __  / / / / /| | / /_/ / / / /
 / /_/ / ___ |/ _, _/ / / / /_/ / /_/ / ___ |/ _, _/ /_/ / 
/_____/_/  |_/_/ |_| /_/ /_____/\____/_/  |_/_/ |_/_____/  
                                                           
      A Cross-Platform Deep Learning Framework.

File: activationSoftmax.dart
Description: Softmax activation function. All outputs sum to 1 
and act as probability/percentages.

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
import 'package:ml_linalg/vector.dart';

///////////////////////////////////////////////////// Class Decl
class ActivationSoftmax {
  //////////////////////// Member Vars
  Matrix? outputs;
  Matrix? doutputs;

  //////////////////////////////// Member Funcs
  Matrix forward(Matrix inputs) {
    List<Iterable<double>> inp = inputs.toList();
    List<List<double>> toReturn = [];
    for (int i = 0; i < inp.length; i++) {
      List<double> working = [];
      List<double> inp2 = inp[i].toList();
      double normBase = 0;
      for (int j = 0; j < inp2.length; j++) {
        double output = exp(inp2[j]);
        working.add(output);
        normBase += output;
      }
      for (int j = 0; j < working.length; j++) {
        working[j] = working[j] / normBase;
      }
      toReturn.add(working);
    }
    return Matrix?.fromList(toReturn);
  }

  List<Matrix> backward(Matrix inputs) {
    List<Iterable<double>> inp = outputs!.toList();
    List<Matrix> toReturn = [];
    for (int i = 0; i < inp.length; i++) {
      List<List<double>> identity = [];
      List<double> inp2 = inp[i].toList();
      print('inp2=');
      print(inp2);
      for (int j = 0; j < inp2.length; j++) {
        List<double> identity2 = [];
        for (int k = 0; k < inp2.length; k++) {
          if (j == k) {
            identity2.add(inp2[j]);
          } else {
            identity2.add(0);
          }
        }
        identity.add(identity2);
      }
      print('identity=');
      print(identity);
      Matrix rowVector = Matrix.fromList([inp2]);
      Matrix columnVector = Matrix.fromColumns([Vector.fromList(inp2)]);
      Matrix worked = columnVector * rowVector;
      Matrix jacobian = Matrix.fromList(identity) - worked;
      print('jacobian=');
      print(jacobian);
      Matrix ans =
          jacobian * Matrix.fromColumns([Vector.fromList(inputs[i].toList())]);
      print('ans=');
      print(ans);
      toReturn.add(ans);
    }
    return toReturn;
  }
}
