/*
    ____  ___    ____  __________  ____  ___    ____  ____ 
   / __ \/   |  / __ \/_  __/ __ )/ __ \/   |  / __ \/ __ \
  / / / / /| | / /_/ / / / / __  / / / / /| | / /_/ / / / /
 / /_/ / ___ |/ _, _/ / / / /_/ / /_/ / ___ |/ _, _/ /_/ / 
/_____/_/  |_/_/ |_| /_/ /_____/\____/_/  |_/_/ |_/_____/  
                                                           
      A Cross-Platform Deep Learning Framework.

File: lossCategoricalCrossEntropy.dart
Description: Loss function commonly used to classify and with one
hot encoding. L=-\sigma_j{y_j*log(y^hat_j)} where y_j is the actual
and y^hat_j is the predicted.

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
class LossCategoricalCrossEntropy {
  //////////////////////// Member Vars
  Matrix? dinputs;

  //////////////////////////////// Member Funcs
  double forward(Matrix predicted, Matrix actual) {
    assert(predicted.columnsNum == actual.columnsNum);
    assert(predicted.rowsNum == actual.rowsNum);
    List<Iterable<double>> p = predicted.toList();
    List<Iterable<double>> a = actual.toList();
    double toReturn = 0;
    for (int i = 0; i < p.length; i++) {
      List<double> p2 = p[i].toList();
      List<double> a2 = a[i].toList();
      for (int j = 0; j < p2.length; j++) {
        toReturn += (a2[j] * log(p2[j]));
      }
    }
    return -toReturn;
  }

  Matrix backward(Matrix dvalues, Matrix y_true) {
    // Number of samples
    int samples = dvalues.length;
    // Number of labels in every sample
    // We'll use the first sample to count them
    int labels = dvalues[0].length;
    // If labels are sparse, turn them into one-hot vector
    List<Iterable<double>> working = y_true.toList();
    Matrix? y;
    if (working[0].length == 1) {
      List<List<double>> newY = [];
      for (int i = 0; i < working.length; i++) {
        int target = working[i].toList()[0].floor();
        List<double> working2 = [];
        for (int j = 0; j < labels; j++) {
          if (target == j) {
            working2.add(-1);
          } else {
            working2.add(0);
          }
        }
        newY.add(working2);
      }
      y = Matrix.fromList(newY);
    } else {
      List<List<double>> newY = [];
      for (int i = 0; i < working.length; i++) {
        List<double> working2 = working[i].toList();
        for (int j = 0; j < working2.length; j++) {
          working2[j] = -working2[j];
        }
        newY.add(working2);
      }
      y = Matrix.fromList(newY);
    }
    // Calculate gradient
    dinputs = y / dvalues;
    // Normalize gradient
    dinputs = dinputs! / samples;
    return dinputs!;
  }
}
