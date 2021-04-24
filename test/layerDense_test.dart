import 'package:dartboard/layerDense.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ml_linalg/matrix.dart';

/*
    ____  ___    ____  __________  ____  ___    ____  ____ 
   / __ \/   |  / __ \/_  __/ __ )/ __ \/   |  / __ \/ __ \
  / / / / /| | / /_/ / / / / __  / / / / /| | / /_/ / / / /
 / /_/ / ___ |/ _, _/ / / / /_/ / /_/ / ___ |/ _, _/ /_/ / 
/_____/_/  |_/_/ |_| /_/ /_____/\____/_/  |_/_/ |_/_____/  
                                                           
      A Cross-Platform Deep Learning Framework.

File: layerDense_test.dart
Description: Tests the dense layer forward pass.

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
  test('test matrix calculations', () {
    final ld = LayerDense.testMatrixCalcs(
        weights: Matrix.fromList([
          [0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]
        ]).transpose(),
        biases: Matrix.fromList([
          [2.0, 3.0, 0.5],
          [2.0, 3.0, 0.5],
          [2.0, 3.0, 0.5]
        ]));
    final ans = ld.forward(Matrix.fromList([
      [1.0, 2.0, 3.0, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]
    ]));
    expect(
        ans,
        Matrix.fromList([
          [4.800000190734863, 1.209999918937683, 2.384999990463257],
          [8.899999618530273, -1.8100004196166992, 0.19999998807907104],
          [1.4100000858306885, 1.0509999990463257, 0.025999903678894043]
        ]));
  });

  test('test with nonrandom initial weights', () {
    final ld = LayerDense.withInitNeuron(4, 3, 0.5, batchSize: 3);
    final ans = ld.forward(Matrix.fromList([
      [1.0, 2.0, 3.0, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]
    ]));

    final ldExpect = LayerDense.testMatrixCalcs(
        weights: Matrix.fromList([
          [0.5, 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 0.5],
          [0.5, 0.5, 0.5, 0.5]
        ]).transpose(),
        biases: Matrix.fromList([
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]
        ]));
    final ansExpect = ldExpect.forward(Matrix.fromList([
      [1.0, 2.0, 3.0, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]
    ]));
    expect(ans, ansExpect);
  });

  test('test output sizes match neurons and multilayers', () {
    final ld1 = LayerDense(4, 3, batchSize: 3);
    final ans1 = ld1.forward(Matrix.fromList([
      [1, 2, 3, 2.5],
      [2.0, 5.0, -1.0, 2.0],
      [-1.5, 2.7, 3.3, -0.8]
    ]));
    expect(ans1.columnsNum, 3); // 3 = numNeurons
    expect(ans1.rowsNum, 3); // 3 = batch size
    final ld2 = LayerDense(3, 2, batchSize: 3);
    final ans2 = ld2.forward(ans1);
    expect(ans2.columnsNum, 2); // 2 == numNeurons
    expect(ans2.rowsNum, 3); // 3 == batchSize
  });
}
