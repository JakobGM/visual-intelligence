# Model contributions

## T3: CNNs & ImClass

### AlexNet

### VGG

* Insight in depth, small kernels, lots of filters.

### Inception-v1 / GoogLeNet

* Auxiliary networks helps with vanishing gradients.
* Stacking (1, 1) convolution with (5, 5) convolution reduces computational cost. This is combined with filter concatenation.
* The (1, 1) convolution reduces the number of channels before the more complex convolutions.

### EfficientNet

* Deeper and wider networks and greater image resolution for better performance.
* Compound scaling: determine ratio between depth, width, and resolution using a grid search.
* Finds the best way to scale up a network, maximizing the utility of extra complexity.

### CapsuleNets

### RNN/LSTM

* Captures temporal relationships.
* Handles variable-length sequences, long-term dependencies, order information, and cross-sequence parameter sharing.
* LSTM solves the vanishing gradient problem, resulting in uninterrupted gradient flow.

### Deep residual Learning for Image Recognition (ResNet)

* Submissions are getting better and better when it gets better
* Residual block to prevent vanishing gradients
* Shortcut skipping in order to increase depth
* Works because shortcuts allows residuals to flow in network

## T4: Object Detection

### Object Detection (generally)

* Overfeat: sliding image to solve multiple instances

### Object Detection Metrics

* Classification vs. localization
* Recall does not care about false positives, while precision does
* Precision-recall curve
* Average precision
* mAP - mean average precision
* IoU - Intersection over Union

### YOLO v1/v2/v3

* Real-time object detector
* Released 2016
* State-of-the-art in 2016
  * Deformable Parts Models (DPM): sliding windows, no information sharing
  * R-CNN
* "You Only Look Once"
* 25 ms latency (much faster)
* Contextualizes better
* Generalizes better
* Inspired by GoogLeNet, 25 conv layers
* First version struggles with localization (twice as much error as Fast R-CNN), and small objects
* Yolo v2:
  * Better: batch normalization, different architecture, pass-through layers (fine-grained features, small objects), training on images in different resolutions
  * Faster: new architecture (Darknet-19), fewer operations for a forward pass
  * Stronger: can classify many different objects, object detection datasets are usually much smaller (combines classification and detection datasets), gives more specific classifications
* Yolo v3:
  * Bunch of small changes that makes it better: hyperparameters, better bounding box prediction
  * New model: Darknet 53
  * Contributions: changed base architecture to make it more accurate

### YOLO part 2

* Fast YOLO has 9 conv layers instead of 24
* Anchor boxes reduced accuracy but increased recall in YOLO v2
  * The grid went to odd number of grid cells in order to center large objects in the middle of the image

### Region-based CNN method (R-CNN)

* Introduced not long after AlexNet, and used that as a base
* They used classical image processing methods in order to find regions of interest first, it is a two-stage method
* The second stage consists of warping the areas of interest into a pre-defined shape and send that region into a neural network, treating each region as a classification problem
* Proposal stage -> detection stage
* First step that proposes 2.000 regions, and then the image goes through the convolutional pipeline.

### Fast R-CNN

* Slow R-CNN was really slow due to the feature extraction for every one of the 2.000 regions. A lot of unnecessary work.
* Solved by putting the complete image through the convolutional layer, extracting features. This results in a feature map that can be reused.
* Region of interest pooling was introduced in order to make this work, solving the regions of interest problem.
* Still arbitrary sized region of interest proposal. The regions are taken into the feature map, and fixed feature maps are extracted. 

### Faster R-CNN

* The region proposal network got rid of the classical region of interest proposal algorithm. It got replaced by a neural network. It is two networks connected together. The selective search classical method could therefore be removed. They were trained separately and combined after the fact.
* Later, different resolution feature maps where introduced in a pyramid layer.
* Region proposal network is the main contribution.

### SSD: Single Shot MultiBox Detector

* YOLO, Faster R-CNN was the state of the art: region proposals, fully connected layers
* Offered speed and accuracy improvements
* Uses VGG-16 as its base for layers 1-5

### R-FCN & RetinaNet

* Regional fully connected networks
  * Nothing is trainable up until the point where regions of interests are extracted
  * Position Sensitive Score Maps is the main contribution
  * Main contribution: encode position sensitivity for the result in score maps in order to force the region to overlap the object well. When the sub-area count increases, the more correct it becomes. Every class has multiple score maps for different parts of the object.
* RetinaNet
  * The loss function is of interest
  * Focal loss
  * Main contribution: Not the architecture, but the loss function that weights the easy examples down and hard problems up in order to learn more from hard examples in order to improve more

### Cascading R-CNN

* State of the art!
* Usually regressor (find bounding box) and detector (classify object) in most R-CNN model architectures
  * Problem: regressors usually perform best at the level they were trained at. Different thresholds therefore requires training under the same threshold.
* Contributions: selves the overfitting problem when using IoU >> 0.5
* Gives better performance on almost all two-stage models with marginal computation and memory increase
* Much better performance than state of the art single-models
