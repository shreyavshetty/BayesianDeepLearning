# BayesianDeepLearning
Overview of the following papers published on Bayesian Deep Learning:
1. Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding
2. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
3. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

## Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding

- Author : Alex Kendall,Vijay Badrinarayanan,Roberto Cipolla - University of Cambridge
- Published : 10 Oct 2016
- Link : [Paper](https://arxiv.org/pdf/1511.02680.pdf)
### Aim
This paper aims to provide a probabilistic approach to pixel-wise segmentation - SegNet. This paper also aims to extend deep convolutional encoder-decoder neural network architecture to Bayesian CNN which can produce a probabilistic segmentation output. 
### Result
Modelling uncertainity improved the performance by 2-3% compared to the state-of-art architectures like SegNet, FCN and Dilatio Network. Significant improvement in performance was observered when fed with smaller datasets.
### Overview
**Pixel wise Semantic Segmentation -**
Segmentation in Computer Vision implies  partitioning of image into coherent parts without understanding what each component represents. Semantic Segmentation in particular implies that the images are partitioned into semantically meaningful components. When the same goal is achieved by classifying each pixel then it is termed as pixel wise semantic segmentation. 

**Related Work**
- Non Deep Learning Approaches
    - TextonBoost
    - TextonForest
    - Random Forest Based Classifiers
- Deep Learning Approaches - Core Segmentation Engine
    - SegNet
    - Fully Convolutional Networks (FCN)
    - Dilation Network
None of the above methods provide produce a probabilistic segmentation with a measure of model uncertainty.
- Bayesian Deep Learning Approaches
    - Bayesian neural networks [Paper1](https://papers.nips.cc/paper/419-transforming-neural-net-output-levels-to-probability-distributions.pdf) [Paper2](https://authors.library.caltech.edu/13793/). They offer a probabilistic interpretation of deep learning models by inferring distributions over the networks weights. They are often computationally very expensive, increasing the number of model
parameters without increasing model capacity significantly.
    - Performing inference in Bayesian neural networks is a difficult task, and approximations to the model posterior are
often used, such as variational inference [Paper](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)
    - Training with stochastic gradient descent, using dropout to randomly remove units. During test time, standard dropout approximates the effect of averaging the predictions of all these thinnned networks by using the weights of the unthinned network. This is referred to as weight averaging.[Paper](http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)
    - Dropout as approximate Bayesian inference over the network’s weights. [Paper](https://arxiv.org/pdf/1506.02158.pdf)

**SegNet Architecture-** Since the paper aims to provide bayesian approach to the existing SegNet Architecture, the following gives a brief overview of the architecture itself [Paper](https://arxiv.org/pdf/1511.00561) [Blog](https://saytosid.github.io/segnet/):
![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/SegNet_Architecture.png "SegNet Architecture")
- SegNet is a deep convolutional encoder decoder architecture which consists of a sequence of non-linear processing layers (encoders) and a corresponding set of decoders followed by a pixel-wise classifier.
- Encoder consists of one or more convolutional layers with batch normalisation and a ReLU non-linearity, followed by non-overlapping max-pooling and sub-sampling. The sparse encoding due to the pooling is upsampled in the decoder using max-pooling indices in the encoding sequences. It helps in retaining the class boundary details in the segmented images and also reducing the total number of model parameters.
- Encoder Architecture : 
![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/encoder_explained.png "Encoder Architecture")
    - 13 VGG16 Conv layers
    - Not fully connected(this reduces parameters) 
    - Good initial weights are available 
    - Max-pooling and Subsampling - translation invariance achieved but feature map size reduces leading to lossy image representation with blurred boundaries. Hence, upsampling is done in the decoder. 
- Decoder Architecture : 
    - For each of the encoders there is a corresponding decoder which upsamples the feature map using memorised max-pooling indices.To do that it needs to store some information. It is necessary to capture and store boundary information in the encoder feature maps before sub-sampling. In order to to that space efficiently, SegNet stores only the max-pooling indices i.e. the locations of maximum feature value in each pooling window is memorised for each encoder map. Only 2 bits are needed for each window of 2x2, slight loss of precision, but tradeoff.
![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/upsampling_indices.png "Upsampling Index")
    - Sparse feature maps of higher resolutions produced
    - Sparse maps are fed through a trainable filter bank to produce dense feature maps
    - The last decoder is connected to a softmax classifier which classifies each pixel
For each of the 13 encoders there is a corresponding decoder. The model is trained end to end using stochastic gradient descent. 

**Key aspects of Bayesian SegNet Architecture**

- Necessity to find the posterior distribution over the convolutional weights, W, given observed training data X and labels Y. p(W | X, Y). This is difficult to trace. Hence, approximate using variational inference. 
- Let q(W) be the distribution over the network's weights, minimizing the Kullback-Leibler (KL) divergence between this approximating distribution and the full posterior: KL(q(W) || p(W | X, Y)) [Paper](https://arxiv.org/pdf/1506.02158.pdf)
- Using stochastic gradient descent, minimizes the divergence term.
- Use dropouts to form a probabilistic encoder-decoder architecture. This is kept as 0.5. Sample the posterior distribution over the weights at test time using dropout to obtain the posterior distribution of softmax class probabilities. Take the mean of these samples for segmentation prediction and use the variance to output model uncertainty for each class. The mean of the per class variance measurements as an overall measure of model uncertainty. 
- Probabilistic Variants of this architecture :
    - Bayesian Encoder: insert dropout after each encoder unit.
    - Bayesian Decoder: insert dropout after each decoder unit.
    - Bayesian Encoder-Decoder: insert dropout after each encoder and decoder unit.
    - Bayesian Center: insert dropout after the deepest encoder, between the encoder and decoder stage.
    - Bayesian Central Four Encoder-Decoder: insert dropout after the central four encoder and decoder units.
    - Bayesian Classifier: insert dropout after the last decoder unit, before the classifier.
  
  **Comparing Weight Averaging and Monte Carlo Dropout Sampling**
  
  Weight averaging proposes to remove dropout at test time and scale the weights proportionally to the dropout percentage. Monte Carlo sampling with dropout performs better than weight averaging after approximately 6 samples. Weight
averaging technique produces poorer segmentation results,in terms of global accuracy, in addition to being unable to
provide a measure of model uncertainty. 







