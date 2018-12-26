# **Traffic Sign Recognition project** 

### The present work describes the rationale and design decisions behind the development of a deep convolutional neural network architecture to classify german traffic signs 

## Table of Contents

<!-- MarkdownTOC autolink="true" bracket="round"-->

- [1. Data set summary and exploration](#1.-data-set-summary-and-exploration)
  - [a. Load the data set and summarize](#a.-load-the-data-set-and-summarize)
  - [b. Explore and visualize the data set](#b.-explore-and-visualize-the-data-set)
- [2. Design, train, and test a model architecture](#2.-design,-train,-and-test-a-model-architecture)
  - [a. Image preprocessing and augmentation](#1-image-preprocessing-and-augmentation)
  - [b. Final model architecture](#2-final-model-architecture)
  - [c. Model training](#3-model-training)
  - [d. Final results and discussion](#4-final-results-and-discussion)
- [3. Test a model on new images](#test-a-model-on-new-images)
  - [1. Custom traffic sign images](#1-custom-traffic-sign-images)
  - [2. Model performace on the custom images](#2-model-performace-on-the-custom-images)
- [4. Explore the feature maps and the response of different layers of stimuli](#4.-explore-the-feature-maps-and-the-response-of-different-layers-of-stimuli)
<!-- /MarkdownTOC -->


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
## 1. Data set summary and exploration

### **a. Load the data set and summarize**

I used python with the len method and the np.ndarray property shape to get an overview of sizes and shapes for the data set.

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32x32x3 (32 times 32 pixels in 3 different color channels
 R, G and B)
* The number of unique classes/labels in the data set is 43. 

Here is the definition of classes according to the assigned index:

Class ID | Sign name
--- | ---
0|Speed limit (20km/h)
1|Speed limit (30km/h)
2|Speed limit (50km/h)
3|Speed limit (60km/h)
4|Speed limit (70km/h)
5|Speed limit (80km/h)
6|End of speed limit (80km/h)
7|Speed limit (100km/h)
8|Speed limit (120km/h)
9|No passing
10|No passing for vehicles over 3.5 metric tons
11|Right-of-way at the next intersection
12|Priority road
13|Yield
14|Stop
15|No vehicles
16|Vehicles over 3.5 metric tons prohibited
17|No entry
18|General caution
19|Dangerous curve to the left
20|Dangerous curve to the right
21|Double curve
22|Bumpy road
23|Slippery road
24|Road narrows on the right
25|Road work
26|Traffic signals
27|Pedestrians
28|Children crossing
29|Bicycles crossing
30|Beware of ice/snow
31|Wild animals crossing
32|End of all speed and passing limits
33|Turn right ahead
34|Turn left ahead
35|Ahead only
36|Go straight or right
37|Go straight or left
38|Keep right
39|Keep left
40|Roundabout mandatory
41|End of no passing
42|End of no passing by vehicles over 3.5 metric tons


### **b. Explore and visualize the data set** 

#### Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

#### 2. Design, train and test a model architecture

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


