# Model contributions

## T3: CNNs & ImClass

### AlexNet

### VGG

* Insight in depth, small kernels, lots of filters.

### Inception-v1 / GoogLeNet

* Auxiliary networks helps with vanishing gradients.
* Stacking (1, 1) convolution with (5, 5) convolution reduces computational cost. This is combined with filter concatenation.
* The (1, 1) convolution reduces the number of channels before the more complex convolutions.

### ResNet

### EfficientNet

* Deeper and wider networks and greater image resolution for better performance.
* Compound scaling: determine ratio between depth, width, and resolution using a grid search.
* Finds the best way to scale up a network, maximizing the utility of extra complexity.

### CapsuleNets

### RNN/LSTM

* Captures temporal relationships.
* Handles variable-length sequences, long-term dependencies, order information, and cross-sequence parameter sharing.
* LSTM solves the vanishing gradient problem, resulting in uninterrupted gradient flow.
