# BayesianDeepLearning

Reference : Machine Learning: A Probabilistic Perspective,Kevin Murphy

Two ways to look at probability :
- Frequentist Interpretation : It represents long run of events. Eg - Flip the coin many times - lands head half the time.
- Bayesian Interpretation : It quantifies the uncertainity about something - related to information rather than number of trails. Eg - Coin is equally likely to land head/toss on the next toss.

Bayesian Approach has an advantage as it helps in modelling uncertainities of an event that does not occur frequenty. For example, probabaility of melting of ice at 2020 CE. This event ocurrs either once or never. Hence, does not occur frequently. This probabaility indicated how certain/uncertain is it to state that ice will melt at 2020 CE.

Overview of Probability Theory :
Ramdom Variable, say X, is a variable whose possible values are numerical outcomes of a random phenomenon. 
P(X) -> Probabaility that the event X is true.
P(~X) -> Probabaility that the event not X. 1-P(X)
Two types of Random Variables :
- Discrete Random :  A random variable that takes only a countable number of distinct values
P(X=x) -> PMF -> Probability Mass Function
- Continuous Random : A random variable that takes infinite number of possible value
P(X=x) -> PDF -> Probability Density Function

Fundamental Rules:
- Union of 2 Events: 
<a href="https://www.codecogs.com/eqnedit.php?latex=P(A\cup&space;B)&space;=&space;P(A)&space;&plus;&space;P(B)&space;-&space;P(A\bigcap&space;B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A\cup&space;B)&space;=&space;P(A)&space;&plus;&space;P(B)&space;-&space;P(A\bigcap&space;B)" title="P(A\cup B) = P(A) + P(B) - P(A\bigcap B)" /></a>

     - If two events are independent of each other then: 
<a href="https://www.codecogs.com/eqnedit.php?latex=P(A\cup&space;B)&space;=&space;P(A)&space;&plus;&space;P(B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A\cup&space;B)&space;=&space;P(A)&space;&plus;&space;P(B)" title="P(A\cup B) = P(A) + P(B)" /></a>

- Joint Probabilities :
<a href="https://www.codecogs.com/eqnedit.php?latex=P(A,B)&space;=&space;P(A\bigcap&space;B)&space;=&space;P(A|B)&space;*&space;P(B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A,B)&space;=&space;P(A\bigcap&space;B)&space;=&space;P(A|B)&space;*&space;P(B)" title="P(A,B) = P(A\bigcap B) = P(A|B) * P(B)" /></a>

Given joint distribution on two events P(A,B), we define the marginal distribution as follows :
<a href="https://www.codecogs.com/eqnedit.php?latex=P(A)&space;=&space;\sum&space;P(A,B)&space;=&space;\sum&space;P(A|B)&space;*&space;P(B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A)&space;=&space;\sum&space;P(A,B)&space;=&space;\sum&space;P(A|B)&space;*&space;P(B)" title="P(A) = \sum P(A,B) = \sum P(A|B) * P(B)" /></a>

Conditional Probability of event A given event B is true, as follows :
<a href="https://www.codecogs.com/eqnedit.php?latex=P(A|B)&space;=&space;P(A,B)/P(B),&space;if&space;P(B)&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A|B)&space;=&space;P(A,B)/P(B),&space;if&space;P(B)&space;>&space;0" title="P(A|B) = P(A,B)/P(B), if P(B) > 0" /></a>

Bayes Rule : <a href="https://www.codecogs.com/eqnedit.php?latex=P(A|B)&space;=&space;\frac{P(A)*P(B|A)}{P(B)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A|B)&space;=&space;\frac{P(A)*P(B|A)}{P(B)}" title="P(A|B) = \frac{P(A)*P(B|A)}{P(B)}" /></a>

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

##  What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
- Author : Alex Kendall,Yarin Gal - University of Cambridge
- Published : 5 Oct 2017
- Link : [Paper](https://arxiv.org/pdf/1703.04977)
### Aim
- Bayesian Framework combining aleatoric uncertainty and epistemic uncertainty.
- New loss functions that captures uncertainity for segmentation and depth regression benchmarks tasks - gives state-of-art results
- Improve model performance by 1 − 3% over non-Bayesian baselines by reducing the effect of noisy data with the implied attenuation
### Overview
Quantifying uncertainty in computer vision applications can be largely divided into regression set-
tings such as depth regression, and classification settings such as semantic segmentation. Two main types of uncertainty:
- Aleatoric uncertainty captures noise inherent in the observations. This could be for example sensor noise or motion
noise, resulting in uncertainty which cannot be reduced even if more data were to be collected.This is modelled by placing a distribution over the output of the model.
    - homoscedastic uncertainty- uncertainty which stays constant for different inputs.Homoscedastic regression assumes constant observation noise σ for every input point x. 
    - heteroscedastic uncertainty - uncertainty depends on the inputs to the model. The observation noise can vary with input x.
      Equation:
		
      Variational inference is performed over the the model output - MAP Inference. learned loss attenuation – making the loss more robust to noisy data.
- Epistemic uncertainty accounts for uncertainty in the model parameters. It captures ignorance about which model generated our collected data. This uncertainty can be explained away given enough data, and is often referred to as model uncertainty.  This is modelled by placing a prior distribution over a model’s weights, and then trying to capture how much these weights vary given some data. 
We are required to evaluate the posterior p(W|X, Y) = p(Y|X, W)p(W)/p(Y|X),cannot be evaluated analytically. Hence, we approximate it.This replaces the intractable problem of averaging over all weights in the BNN with an optimisation task, where we seek to optimise over the parameters of the simple distribution instead of optimising the original neural network’s parameters. We also make use of Dropout variational inference to approximate the models. This inference is done by training a model with dropout before every weight layer,and by also performing dropout at test time to sample from the approximate posterior (stochastic forward passes, referred to as Monte Carlo dropout). More formally, this approach is equivalent to performing approximate variational inference where we find a simple distribution in a tractable family which minimises the Kullback-Leibler (KL) divergence to the true model posterior p(W|X, Y). Dropout can be interpreted as a variational Bayesian approximation.
- Classification Equation: Monte Carlo integration 
- Regression Equation: Uncertainty is captured by the predictive variance. Two terms:
			- first term - sigma sq indicates the noise inherent in the data
			- second term - how much the model is uncertain about its predictions – this term will vanish when we have zero parameter uncertainty
**Combining Aleatoric and Epistemic Uncertainty in One Model**
We can use a single network to transform the input x, with its head split to predict both ŷ as well as σ̂ 2. This loss consists of two components - the residual regression obtained with a stochastic sample through the model – making use of the uncertainty over the parameters – and an uncertainty regularization term.
Heteroscedastic Uncertainty as Learned Loss Attenuation - this makes the model more robust to noisy data: inputs for which the model learned to predict high uncertainty will have a smaller effect on the loss. The model is discouraged from predicting high uncertainty for all points – in effect ignoring the data – through the log σ 2 term. Large uncertainty increases the contribution of this term, and in turn penalizes the model: The model can learn to ignore the data – but is penalised for that. The model is also discouraged from predicting very low uncertainty for points with high residual error, as low σ 2 will exaggerate the contribution of the residual and will penalize the model. It is important to stress that this learned attenuation is not an ad-hoc construction, but a consequence of the probabilistic
interpretation of the model.It is important to stress that this learned attenuation is not an ad-hoc construction, but a consequence of the probabilistic interpretation of the model. 
### Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

