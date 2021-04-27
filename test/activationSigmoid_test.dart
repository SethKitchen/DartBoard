import 'package:dartboard/activations/activationSigmoid.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ml_linalg/matrix.dart';

/*
    ____  ___    ____  __________  ____  ___    ____  ____ 
   / __ \/   |  / __ \/_  __/ __ )/ __ \/   |  / __ \/ __ \
  / / / / /| | / /_/ / / / / __  / / / / /| | / /_/ / / / /
 / /_/ / ___ |/ _, _/ / / / /_/ / /_/ / ___ |/ _, _/ /_/ / 
/_____/_/  |_/_/ |_| /_/ /_____/\____/_/  |_/_/ |_/_____/  
                                                           
      A Cross-Platform Deep Learning Framework.

File: activationSigmoid_test.dart
Description: Tests the sigmoid activation forward pass.

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
void main() {
  test('test calculations', () {
    final aSig = ActivationSigmoid();
    final ans = aSig.forward(Matrix.fromList([
      [1.0, 2.0, 3.0, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]
    ]));
    expect(
        ans,
        Matrix.fromList([
          [
            0.7310585975646973,
            0.8807970881462097,
            0.9525741338729858,
            0.9241418242454529
          ],
          [
            0.8807970881462097,
            0.9933071732521057,
            0.2689414322376251,
            0.8807970881462097
          ],
          [
            0.18242552876472473,
            0.9370266199111938,
            0.9644287824630737,
            0.31002551317214966
          ]
        ]));
  });
}
