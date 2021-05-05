import 'package:dartboard/activations/activationSoftmax.dart';
import 'package:dartboard/losses/lossCategoricalCrossEntropy.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ml_linalg/matrix.dart';

/*
    ____  ___    ____  __________  ____  ___    ____  ____ 
   / __ \/   |  / __ \/_  __/ __ )/ __ \/   |  / __ \/ __ \
  / / / / /| | / /_/ / / / / __  / / / / /| | / /_/ / / / /
 / /_/ / ___ |/ _, _/ / / / /_/ / /_/ / ___ |/ _, _/ /_/ / 
/_____/_/  |_/_/ |_| /_/ /_____/\____/_/  |_/_/ |_/_____/  
                                                           
      A Cross-Platform Deep Learning Framework.

File: activationSoftmax_test.dart
Description: Tests the softmax activation forward pass.

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
  test('test 1D calculations', () {
    final aSig = ActivationSoftmax();
    final ans = aSig.forward(Matrix.fromList([
      [4.8, 1.21, 2.385]
    ]));
    expect(
        ans,
        Matrix.fromList([
          [0.8952826857566833, 0.024708302691578865, 0.08000901341438293]
        ]));
  });

  /*test('test 1-D backward pass', () {
    final aSig = ActivationSoftmax();
    final ans = aSig.backward(Matrix.fromList([
      [0.7, 0.1, 0.2]
    ]));
    expect(
        ans,
        Matrix.fromList([
          [-0.1, 0.03333333, 0.06666667]
        ]));
  });*/

  test('test cross entropy wth softmax', () {
    final softmax_outputs = Matrix.fromList([
      [0.7, 0.1, 0.2],
      [0.1, 0.5, 0.4],
      [0.02, 0.9, 0.08]
    ]);
    final class_targets = Matrix.fromList([
      [0, 1, 1]
    ]);
    final activation = ActivationSoftmax();
    activation.outputs = softmax_outputs;
    final loss = LossCategoricalCrossEntropy();
    final dinputs = loss.backward(softmax_outputs, class_targets);
    print('DINPUTS:::::');
    print(dinputs);
    final ans = activation.backward(dinputs);
    expect(
        ans,
        Matrix.fromList([
          [-0.09999999, 0.03333334, 0.06666667],
          [0.03333334, -0.16666667, 0.13333334],
          [0.00666667, -0.03333333, 0.02666667]
        ]));
  });

  test('test n-D calculations', () {
    final aSig = ActivationSoftmax();
    final ans = aSig.forward(Matrix.fromList([
      [4.8, 1.21, 2.385],
      [8.9, -1.81, 0.2],
      [1.41, 1.051, 0.026]
    ]));
    expect(
        ans,
        Matrix.fromList([
          [0.8952826857566833, 0.024708302691578865, 0.08000901341438293],
          [0.9998111128807068, 0.000022316406102618203, 0.0001665544114075601],
          [0.5130971670150757, 0.35833391547203064, 0.12856893241405487]
        ]));
  });
}
