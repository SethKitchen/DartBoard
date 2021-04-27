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
}
