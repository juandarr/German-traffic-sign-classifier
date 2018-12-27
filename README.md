# **Traffic Sign Recognition project** 

### The present work describes the rationale and design decisions behind the development of a deep convolutional neural network architecture to classify german traffic signs 

## Table of Contents

<!-- MarkdownTOC autolink="true" bracket="round"-->

- [1. Data set summary and exploration](#1-data-set-summary-and-exploration)
  - [a. Load the data set and summarize](#a-load-the-data-set-and-summarize)
  - [b. Explore and visualize the data set](#b-explore-and-visualize-the-data-set)
- [2. Design, train, and test a model architecture](#2-design,-train,-and-test-a-model-architecture)
  - [a. Data augmentation and image preprocessing](#a-data-augmentation-and-image-preprocessing)
  - [b. Final model architecture](#b-final-model-architecture)
  - [c. Model training](#c-model-training)
  - [d. Final results and discussion](#d-final-results-and-discussion)
- [3. Test a model on new images](#3-test-a-model-on-new-images)
  - [1. Custom traffic sign images](#1-custom-traffic-sign-images)
  - [2. Model performace on the custom images](#2-model-performace-on-the-custom-images)
- [4. Explore the feature maps and the response of different layers of stimuli](#4-explore-the-feature-maps-and-the-response-of-different-layers-of-stimuli)
<!-- /MarkdownTOC -->


[//]: # (Image References)

[image1]: ./images/overview_training_images.png "Visualization"
[image2]: ./images/class_distribution.png "Class distribution"
[image3]: ./images/class_0.png "Class 0"
[image4]: ./images/class_1.png "Class 1"
[image5]: ./images/class_2.png "Class 2"
[image6]: ./images/class_3.png "Class 3"
[image7]: ./images/class_4.png "Class 4"
[image8]: ./images/class_5.png "Class 5"
[image9]: ./images/class_6.png "Class 6"
[image10]: ./images/class_7.png "Class 7"
[image11]: ./images/class_8.png "Class 8"
[image12]: ./images/class_9.png "Class 9"
[image13]: ./images/class_10.png "Class 10"
[image14]: ./images/class_11.png "Class 11"
[image15]: ./images/class_12.png "Class 12"
[image16]: ./images/class_13.png "Class 13"
[image17]: ./images/class_14.png "Class 14"
[image18]: ./images/class_15.png "Class 15"
[image19]: ./images/class_16.png "Class 16"
[image20]: ./images/class_17.png "Class 17"
[image21]: ./images/class_18.png "Class 18"
[image22]: ./images/class_19.png "Class 19"
[image23]: ./images/class_20.png "Class 20"
[image24]: ./images/class_21.png "Class 21"
[image25]: ./images/class_22.png "Class 22"
[image26]: ./images/class_23.png "Class 23"
[image27]: ./images/class_24.png "Class 24"
[image28]: ./images/class_25.png "Class 25"
[image29]: ./images/class_26.png "Class 26"
[image30]: ./images/class_27.png "Class 27"
[image31]: ./images/class_28.png "Class 28"
[image32]: ./images/class_29.png "Class 29"
[image33]: ./images/class_30.png "Class 30"
[image34]: ./images/class_31.png "Class 31"
[image35]: ./images/class_32.png "Class 32"
[image36]: ./images/class_33.png "Class 33"
[image37]: ./images/class_34.png "Class 34"
[image38]: ./images/class_35.png "Class 35"
[image39]: ./images/class_36.png "Class 36"
[image40]: ./images/class_37.png "Class 37"
[image41]: ./images/class_38.png "Class 38"
[image42]: ./images/class_39.png "Class 39"
[image43]: ./images/class_40.png "Class 40"
[image44]: ./images/class_41.png "Class 41"
[image45]: ./images/class_42.png "Class 42"
[image46]: ./images/class_augmented.png "Class 36 augmented"
[image47]: ./images/class_distribution_augmentation.png "Class distribution augmentation"
[image48]: ./images/original_image_preprocessing.png "Original image before preprocessing"
[image49]: ./images/yuv_color_space.png "YUV color space transformation"
[image50]: ./images/preprocessing_output.png "Preprocessing output"
[image51]: ./images/preprocessing_output_histograms.png "Preprocessing output histograms"
[image52]: ./images/samples_augmentation_grayscale.png "Some samples after augmentation and grayscale method"
[image53]: ./images/samples_augmentation_ychannel.png "Some samples after augmentation and Y channel method"
[image54]: ./images/custom_test_images.png "Custom test images"
[image55]: ./images/Custom_test_images_preprocessing.png "Custom test images preprocessed"
[image56]: ./images/predictions_custom_images.png "Top 5 predictions of custom set"
[image57]: ./images/stimuli.png "Stimuli"
[image58]: ./images/feature_maps_conv1.png "Feature maps of 1st convolutional layer"
[image59]: ./images/feature_maps_conv2.png "Feature maps of 2nd convolutional layer"


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


Here is an exploratory visualization of the data set.

![alt text][image1]

The code randomly selects images from the training set and plot them in the canvas. You can see the image and the label that defines it.

The following chart shows the class distribution among the total set of training images.

![alt text][image2]

As you can see, a lot of classes have a sample size of less than 500. This is problematic since compared to other classes with 1000 or more samples, that class will get less representative data to train a robust classifier able to discriminate it in scenarios where inputs are different from those in the samples. One way to solve is data augmentation, which will use to multiply the number of representative data for those classes, and will be used in all classes to create images with significant variations in lightness, image position, image scale and image rotation. 

Finally, lets explore the images representing each class. An exploration of each class will allows us to get an idea of the quality of the data and the kind of transformations we could apply to improve the performance of the training/testing pipelines. 

#### **Label 0**
![alt text][image3]
#### **Label 1**
![alt text][image4]
#### **Label 2**
![alt text][image5]
#### **Label 3**
![alt text][image6]
#### **Label 4**
![alt text][image7]
#### **Label 5**
![alt text][image8]
#### **Label 6**
![alt text][image9]
#### **Label 7**
![alt text][image10]
#### **Label 8**
![alt text][image11]
#### **Label 9**
![alt text][image12]
#### **Label 10**
![alt text][image13]
#### **Label 11**
![alt text][image14]
#### **Label 12**
![alt text][image15]
#### **Label 13**
![alt text][image16]
#### **Label 14**
![alt text][image17]
#### **Label 15**
![alt text][image18]
#### **Label 16**
![alt text][image19]
#### **Label 17**
![alt text][image20]
#### **Label 18**
![alt text][image21]
#### **Label 18**
![alt text][image22]
#### **Label 20**
![alt text][image23]
#### **Label 21**
![alt text][image24]
#### **Label 22**
![alt text][image25]
#### **Label 23**
![alt text][image26]
#### **Label 24**
![alt text][image27]
#### **Label 25**
![alt text][image28]
#### **Label 26**
![alt text][image29]
#### **Label 27**
![alt text][image30]
#### **Label 28**
![alt text][image31]
#### **Label 29**
![alt text][image32]
#### **Label 30**
![alt text][image33]
#### **Label 31**
![alt text][image34]
#### **Label 32**
![alt text][image35]
#### **Label 33**
![alt text][image36]
#### **Label 34**
![alt text][image37]
#### **Label 35**
![alt text][image38]
#### **Label 36**
![alt text][image39]
#### **Label 37**
![alt text][image40]
#### **Label 38**
![alt text][image41]
#### **Label 39**
![alt text][image42]
#### **Label 40**
![alt text][image43]
#### **Label 41**
![alt text][image44]
#### **Label 42**
![alt text][image45]

A lot of the images are dark, blurry and have a poor resolution. There is also a mix of objects in several samples and the main objects are portraited from different perspectives. These are the kind of expected images in real scenarios. 

## 2. Design, train and test a model architecture

### **a. Data augmentation and image preprocessing**

#### Data augmentation

Given the class distribution shown in image (2), where some classes have less than 500 samples while other have 1000 or more, it was clear that the augmentation of data was required so we could uniformalize the distribution and give each class an equally significant representation so the classifier can perform well for all classes. To augment the data I decide to use the python library `imgaug`, which can be explore in the following link: [**imgaug**: Image augmentation for machine learning experiments](https://github.com/aleju/imgaug).

There are many augmentation techniques available for us to increase the data set size. From the long list of methods I decided to try the following ones:

* **Cropping**: random crops of percentages of the image between 0 and 10%.
* **Rotation**: random rotation between -15 and 15 degrees.
* **Scaling**: random scaling in both axis with percentages between 80% and 120%.
* **Translation**: random translation in both axis with percentages between -20% and 20%.

The multiple transformations where then used in the whole data set. For classes with sample sizes below 1250, 4 iterations where performed increasing 4 times the data size of each class. Classes with a sample size of more than 1250 where increased with 1000 more samples obtain with the transformation methods. The following image shows some of the new images generated in the data augmentation step:

![alt text][image46]

The images shown are the output of transformation such as rotation, scaling, translation and cropping. For example, the image at the top left has been rotated to the right while the image at the bottom right has been scaled down in the horizontal axis. Adding a new set of images with these transformations will make the training process more powerful, offering a equally distributed sample size for each class and adding variations in terms of translation, scale, rotation and imperfection that will make the classifier more robust.

The following image displays the final class sample size distribution, now almost all classes have a similar sample size.

![alt text][image47]

After the whole process is performed, we change from a total training set of  34799 samples to 100656 samples with an average of about 2400 samples per class. 

#### Preprocessing

Two different preprocessing techniques were designed and implemented in code to preprocess the training data set after augmentation. In the following sections I will describe each method and show visual examples of the output.

##### 1. Grayscaling and normalization

In this method we take the mean values for the pixels in the RGB channels, obtaining one unique pixel matrix with the channel of mean values. After this we apply normalization through the operation `(array-128.0)/128.0`.

The following image shows an original sample and its grayscale representation. 

![alt text][image48]

This image has been translated from its original position in the x and y axis. It also corresponds to the label 1, Speed limit (30km/h). A final output of the grayscale image with normalization will be shown after describing the next method.  

##### 2. Color channel Y and local/global contrast normalization

The original RGB image is transformed to the YUV color space. The channel Y is chosen for preprocessing and the channels U/V are discarded. Local and global contrast normalization are then applied to the Y color channel image. 

After transforming the original RGB image to the YUV color space, we get the following representation of channels.

![alt text][image49]

As we can see, the U and V channels lose a lot of details of the original image. For example, one of the most important patterns of the label -the number 30- is slightly visible. It is for this reason that the Y channel is preffered and the other ones are discarded.

It also of interest to note that the grascale and Y channel versions of the original image are quite similar. The final preprocessing step is the one that makes them differ: relative normalization in the case of the grayscale method (1) and local/global contrast normalization when we pick the Y color channel (2).
This observation is evident when we compare the original image and the final output after applying each method, as shown in the next image.

![alt text][image50]

A comparison of the histograms of the output image after using each method is:

![alt text][image51]

Clearly there is more variability in pixel intensity among the pixels of the Y channel method output. 
The following couple of images show some of the samples of the training set after augmentation and preprocessing using each preprocessing method. 

![alt text][image52]

![alt text][image53]

Experiments were performed with each method separatedly. The best results were obtained with the second method, picking the Y color image and then using local/global contrast normalization.

### **b. Final model architecture**

Here is the final architecture of the deep neural network:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 monocolor image   							| 
| Convolution 3x3     	| 1x1 stride, 'VALID' padding, outputs 30x30x32 	|
| RELU					|												|
| Dropout               | Keep prob = 0.7                               |
| Convolution 3x3     	| 1x1 stride, 'VALID' padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x64 				|
| Dropout               | Keep prob = 0.7                               |
| Convolution 3x3     	| 1x1 stride, 'VALID' padding, outputs 12x12x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x128      			|
| Dropout               | Keep prob = 0.7                               |
| Fully connected		| inputs 4608, outputs 600       				|
| RELU				|       									|
| Dropout               | Keep prob = 0.5                               |
| Fully connected		| inputs 600, outputs 150       				|
| RELU				|       									|
| Dropout               | Keep prob = 0.5                               |
| Logits				| inputs 150, outputs 43						|

This final output is in the form of logits, which are fed to a cost function using the softmax function (mean cross entropy with softmax).

### **c. Model training**

The following table presents the hyperparameters used to train the model:

| Hyperparameter | Value |
| ---------------| --------------|
|EPOCHS          | 100          |
| BATCH_SIZE     | 128          |
| Learning rate  | 0.001        |
| Dropout CNL*   | 0.7          |
| Dropout FCL*     | 0.5          |

`CNL stands for Convolution Neural Layer and FCL for Fully Connected Layer`.

The optimizer used was AdamOptimizer.

### **d. Final results and discussion**

My final model results were:
* training set accuracy of ? 99%
* validation set accuracy of ? 97.3% 
* test set accuracy of ? 95.1%



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

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
 
### **3. Test a model on new images**

#### **1. Custom traffic sign images**

For each image, discuss what quality or qualities might be difficult to classify.

The following figures pack the set of customs images to analyze. The first one correspond to the original RGB image, and the second one shows the images after preprocessing using the Y channel and local/global contrast normalization.

![alt text][image54]

![alt text][image55]

These images are quite clean. From the training , validation and testing performance we should expect to get a classification accuracy close to 100%. Compared with the images from the test set these images are centered, without changes in perspective, with high contrast, the patterns are free noise and clearly identifiable. We'll see how we score in the real test in the following section.  


#### **2. Model performace on the custom images**

(OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The following are the results of the preditions:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road      			|  Priority road 										|
| Stop       		    | Stop     									| 
| Children crossing				| 	Children crossing									|
| Speed limit (60km/h)   		| Speed limit (60km/h) 				 				|
| Road work			|      Road work							|

The model was able to predict the whole custom data set. The accuracy predicting the test set was about 95.1%, so we should expect a close value in the current test, more considering that the images included in this custom test set are an easy target if the deep network has been properly trained. This is the case, the performance is 100%. 

#### Top 5 softmax probabilities of each image in the custom set

To make the prediction of the top 5 softmax probabilities I used the `tf.nn.softmax()` function with `tf.nn.top_k(softmax_values, k = 5)`. After performing these operations (implemented in the jupyter notebook) we get the following representations:

![alt text][image56]

* Verify that the notebook from your laptop and the cloud are the same
* Explain both of the processing techniques you used here
* Explain the different architectures
* Explore other solutions

### **4. Explore the feature maps and the response of different layers of stimuli**

Lets explore the feature maps in the deep neural networks. We have three convolutional layers in the neural network. We know that as we go deeper in the network the number of filters increases and we get access to more complex patterns. Starting with simple geometrical features such as lines, edges and points in the first convolutional layer to circles, squares and curves in the layer that follows and finally, complex patterns such as faces, car/house shapes and more moving deeper in the network. However, since we are using two max pooling layer here and since the original image size is 32x32 pixels, after we apply two of them we will downsize the matrix image to half of the output of the first convolution, and then half of the output of the second convolution, we lose a lot of visual details here that can be discerned with the feature maps. Therefore, we will focus only in the first two convolutions.
Here is the input image we will use as the stimuli:

![alt text][image57]

The following are the feature maps in the first convolutional layer after using the first image of the custom set as stimuli:

![alt text][image58]

and the response after using the same stimuli in the second convolutional layer:

![alt text][image59]

In the feature maps of the first convolutional layer we see the network is sensible to lines. There are a lot of activations in adyacent pixels forming lines in the response. This is a good sign since the input image has lines in it, defining rotated squares. The second convolutional layer response, which has been halved by the max pooling layer, has less resolution , but it appears to have a preference for curves and triangles in the corners. The feature maps of the third convolutional network have a resolution of 6x6 pixels and are not as interesting as the previous maps. I will ingnore them in this discussion.


