# BayesianDeepLearning

Reference : Machine Learning: A Probabilistic Perspective,Kevin Murphy

In machine learning, probability is used to model concepts.
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

Sum Rule : <a href="https://www.codecogs.com/eqnedit.php?latex=P(A)&space;=&space;\sum_{b}&space;P(A,B)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A)&space;=&space;\sum_{b}&space;P(A,B)" title="P(A) = \sum_{b} P(A,B)" /></a>

Product Rule : <a href="https://www.codecogs.com/eqnedit.php?latex=P(A,B)&space;=&space;P(B|A)&space;*&space;P(A)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A,B)&space;=&space;P(B|A)&space;*&space;P(A)" title="P(A,B) = P(B|A) * P(A)" /></a>

Conditional Probability of event A given event B is true, as follows :
<a href="https://www.codecogs.com/eqnedit.php?latex=P(A|B)&space;=&space;P(A,B)/P(B),&space;if&space;P(B)&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A|B)&space;=&space;P(A,B)/P(B),&space;if&space;P(B)&space;>&space;0" title="P(A|B) = P(A,B)/P(B), if P(B) > 0" /></a>

Bayes Rule : <a href="https://www.codecogs.com/eqnedit.php?latex=P(A|B)&space;=&space;\frac{P(A)*P(B|A)}{P(B)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A|B)&space;=&space;\frac{P(A)*P(B|A)}{P(B)}" title="P(A|B) = \frac{P(A)*P(B|A)}{P(B)}" /></a>

This bayesian probability talks about partial beliefs and calculates the validity of a proposition. This calculation is based on :
- Prior Estimate
- New relevant evidence

Probability of a given hypothesis given data -<a href="https://www.codecogs.com/eqnedit.php?latex=P(h|D)&space;=&space;\frac{P(D|h)*P(h)}{P(D)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(h|D)&space;=&space;\frac{P(D|h)*P(h)}{P(D)}" title="P(h|D) = \frac{P(D|h)*P(h)}{P(D)}" /></a>
Goal of Bayes Learning is to achieve the most probabale hypothesis - (MAP)  - <a href="https://www.codecogs.com/eqnedit.php?latex=argmax_{h\sqsubset&space;H}&space;P(h|D)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?argmax_{h\sqsubset&space;H}&space;P(h|D)" title="argmax_{h\sqsubset H} P(h|D)" /></a>
Since P(D) is independent of the hypothesis, this can be eliminated. Also, for all hypothesis P(h) is equal. Hemce, that can be eliminated too. So, the final equation so obtained will involve maximizing the likelihood.
<a href="https://www.codecogs.com/eqnedit.php?latex=h_{ML}&space;=&space;argmax_{h\subset&space;H}&space;P(D|h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{ML}&space;=&space;argmax_{h\subset&space;H}&space;P(D|h)" title="h_{ML} = argmax_{h\subset H} P(D|h)" /></a>

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
- Minimising the cross entropy loss objective function has the effect of minimising the Kullback-Leibler divergence term
- Using stochastic gradient descent, minimizes the divergence term. Therefore training the network with stochastic gradient descent will encourage the model to learn a distribution of weights which explains the data well while preventing overfitting. 
- Use dropouts to form a probabilistic encoder-decoder architecture. This is kept as 0.5. Sample the posterior distribution over the weights at test time using dropout to obtain the posterior distribution of softmax class probabilities. Take the mean of these samples for segmentation prediction and use the variance to output model uncertainty for each class. The mean of the per class variance measurements as an overall measure of model uncertainty.
- We use batch normalisation layers after every convolutional layer
- The whole system is trained end-to-end using stochastic gradient descent with a base learning rate of 0.001 and weight decay parameter equal to 0.0005. We train the network until convergence when we observe no further reduction in training loss.
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

**Experiments**
Bayesian Deep Learning performs well on the following dataset:
- CamVid Dataset
	- CamVid is a small road scene understanding dataset.
	- Segment 11 classes.
	- Comparision based on depth and motion cues.
	- Bayesian SegNet obtains the highest overall class average and mean intersection over union score by a significant margin.
- Scene Understanding (SUN)
	- Large dataset of indoor scenes.
	- 37 indoor scene classes.
	- Bayesian SegNet outperforms all previous benchmarks, including those which use depth modality.
- Pascal VOC
	- Segmentation challenge
	- 20 salient object class
	- Involves learning both classes and their spatial context
	
**Understanding Modelling Uncertainity**
- Qualitative observations
Segmentation predictions are smooth, with a sharp segmentation around object boundaries. These results also show that when the model predicts an incorrect label, the model uncertainty is generally very high. High model uncertainity due to:
	- At class boundaries the model often displays a high level of uncertainty. This reflects the ambiguity surrounding the definition of defining where these labels transition.
	- Visually difficult to identify - objects are occluded or at a distance from the camera.
	- Visually ambiguous to the model.
- Quantitative observation
Relationship between uncertainity vs accuracy and between uncertainity vs frequency of each class in the dataset.Uncertainty is calculated as the mean uncertainty value for each pixel of that class in a test dataset. We observe an inverse  relationship between uncertainty and class accuracy or class frequency. This shows that the model is more confident about classes which are easier or occur more often, and less certain about rare and challenging classes.

 

##  What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?
- Author : Alex Kendall,Yarin Gal - University of Cambridge
- Published : 5 Oct 2017
- Link : [Paper](https://arxiv.org/pdf/1703.04977)
### Aim
- Bayesian Framework combining aleatoric uncertainty and epistemic uncertainty.
- New loss functions that captures uncertainity for segmentation and depth regression benchmarks tasks - gives state-of-art results
- Improve model performance by 1 − 3% over non-Bayesian baselines by reducing the effect of noisy data with the implied attenuation
### Overview
Quantifying uncertainty in computer vision applications can be largely divided into regression settings such as depth regression, and classification settings such as semantic segmentation. Two main types of uncertainty:
- Aleatoric uncertainty captures noise inherent in the observations. This could be for example sensor noise or motion
noise, resulting in uncertainty which cannot be reduced even if more data were to be collected. This is modelled by placing a distribution over the output of the model.
    - homoscedastic uncertainty- uncertainty which stays constant for different inputs. Homoscedastic regression assumes constant observation noise σ for every input point x. 
    - heteroscedastic uncertainty - uncertainty depends on the inputs to the model. The observation noise can vary with input x.
    
- Epistemic uncertainty accounts for uncertainty in the model parameters. It captures ignorance about which model generated our collected data. This uncertainty can be explained away given enough data, and is often referred to as model uncertainty.  This is modelled by placing a prior distribution over a model’s weights, and then trying to capture how much these weights vary given some data. 
We are required to evaluate the posterior p(W|X, Y) = p(Y|X, W)p(W)/p(Y|X),cannot be evaluated analytically. Hence, we approximate it.This replaces the intractable problem of averaging over all weights in the BNN with an optimisation task, where we seek to optimise over the parameters of the simple distribution instead of optimising the original neural network’s parameters. We also make use of Dropout variational inference to approximate the models. This inference is done by training a model with dropout before every weight layer,and by also performing dropout at test time to sample from the approximate posterior (stochastic forward passes, referred to as Monte Carlo dropout). More formally, this approach is equivalent to performing approximate variational inference where we find a simple distribution in a tractable family which minimises the Kullback-Leibler (KL) divergence to the true model posterior p(W|X, Y). Dropout can be interpreted as a variational Bayesian approximation. Epistemic uncertainty in the weights can be reduced by observing more data. This uncertainty induces prediction uncertainty by marginalising over the (approximate) weights posterior distribution. 
- Classification Equation: Monte Carlo integration 

![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/Classification_Eq.png "Classifier_Eq")

- Regression Equation: Uncertainty is captured by the predictive variance.

![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/Regression_Eq.png "Regression_Eq")

Two terms:
			- first term - sigma sq indicates the noise inherent in the data
			- second term - how much the model is uncertain about its predictions – this term will vanish when we have zero parameter uncertainty
**Combining Aleatoric and Epistemic Uncertainty in One Model**
Heteroscedastic equation can be turned into a Bayesian NN by placing a distribution over its weights.
We can use a single network to transform the input x, with its head split to predict both ŷ as well as σ̂ 2. 
![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/Eq.png "Eq")

This loss consists of two components - the residual regression obtained with a stochastic sample through the model – making use of the uncertainty over the parameters – and an uncertainty regularization term. We do not need ‘uncertainty labels’ to learn uncertainty. Rather, we only need to supervise the learning of the regression task. We learn the variance, σ^2 , implicitly from the loss function. The second regularization term prevents the network from predicting infinite uncertainty
(and therefore zero loss) for all data points.In practice, we train the network to predict the log variance. This is because it is more numerically stable than regressing the variance, σ^2 , as the loss avoids a potential division by zero.
![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/Eq3.png "Eq3")


Heteroscedastic Uncertainty as Learned Loss Attenuation - this makes the model more robust to noisy data: inputs for which the model learned to predict high uncertainty will have a smaller effect on the loss. The model is discouraged from predicting high uncertainty for all points – in effect ignoring the data – through the log σ 2 term. Large uncertainty increases the contribution of this term, and in turn penalizes the model: The model can learn to ignore the data – but is penalised for that. The model is also discouraged from predicting very low uncertainty for points with high residual error, as low σ 2 will exaggerate the contribution of the residual and will penalize the model. It is important to stress that this learned attenuation is not an ad-hoc construction, but a consequence of the probabilistic
interpretation of the model.It is important to stress that this learned attenuation is not an ad-hoc construction, but a consequence of the probabilistic interpretation of the model. 

Heteroscedastic Uncertainty in Classification Tasks :For classification tasks our NN predicts a vector of unaries f i for each pixel i, which when passed through a softmax operation, forms a probability vector p i . The model is changed by placing a Gaussian distribution over the unaries vector:
![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/Eq4.png "Eq4")

Monte Carlo integration is uesd and sample unaries through the softmax function. The equation is as follows:
![alt text](https://github.com/shreyavshetty/BayesianDeepLearning/blob/master/Eq5.png "Eq5")

Experiments : 
- Semantic Segmentation 
	- CamVid - intersection over union (IoU) score of 67.5% - The implicit attenuation obtained from the aleatoric loss provides a larger improvement than the epistemic uncertainty model. However, the combination of both uncertainties improves performance even further.
	- NYU Dataset - This dataset is much harder than CamVid because there is significantly less structure in indoor scenes. Improves baseline performance by giving the model flexibility to estimate uncertainty and attenuate the loss.
- Pixel-wise Depth Regression
	
What Do Aleatoric and Epistemic Uncertainties Capture?
- Quality of Uncertainty Metric
	- precision-recall curves for regression and classification models show how model performance improves by removing pixels with uncertainty larger than various percentile thresholds.
	- correlate well with accuracy and precision
	- the curves for epistemic and aleatoric uncertainty models are very similar. This suggests that when only one uncertainty is explicitly modeled, it attempts to compensate for the lack of the alternative uncertainty when possible
	
Uncertainty with Distance from Training Data
- Epistemic uncertainty decreases as the training dataset gets larger.
- Aleatoric uncertainty remains relatively constant and cannot be explained away with more data
- Testing the models with a different test set shows that epistemic uncertainty increases considerably 

# Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
- Author : Alex Kendall,Yarin Gal,Roberto Cipolla
- Published : 24 April 2018
- Link : [Paper](https://arxiv.org/pdf/1511.02680.pdf)
## Aim :
The proposed principled approach to multi-task deep learning weighs multiple loss functions by considering the homoscedastic uncertainty of each task. It simultaneously learn various quantities with different units or scales in both classification and regression settings. 
## Overview :
