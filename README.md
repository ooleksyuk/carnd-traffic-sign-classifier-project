## <a name="top"></a> Deep Learning: Traffic Sign Recognition Classifier [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)

**Traffic Sign Recognition Project** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/32x32/00001.png "Traffic Sign 1"
[image5]: ./test/32x32/00002.png "Traffic Sign 2"
[image6]: ./test/32x32/00003.png "Traffic Sign 3"
[image7]: ./test/32x32/00004.png "Traffic Sign 4"
[image8]: ./test/32x32/00005.png "Traffic Sign 5"
[image9]: ./examples/priview-data-set.png "Data Set Example"
[image10]: ./examples/data-set-histogram.png "Histograms of data set"
[image11]: ./examples/new_images_example.png "New image examples"

## Rubric Points
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.
### Files Submitted:
#### Submission Files: The project submission includes all required files.
My project includes:
* python notebook
* report.html file of the python note book
* examples of images I used to identify signs
* this write up report

### Dataset Exploration:
#### Dataset Summary: The submission includes a basic summary of the data set.
I used data set provided in the project: test and training sets. To get validation set I took 30% of training set.
To get data for sign classification I used German Image data base of signs, per project recommendation
#### Exploratory Visualization: The submission includes an exploratory visualization on the dataset.
I looped through all images provided in data set and displayed them in 4z4 plot with sign name label on top of the individual sign picture
I took 'signnames.csv' and populated a python dictionary with names of signs and built a histograms of signs per class. I used Panda for this task.
### Design and Test a Model Architecture:
#### Preprocessing: The submission describes the preprocessing techniques used and why these techniques were chosen.
Before building me LeNet I processed data of the images. I did this to make the process of training more efficient.
To process images I used `equalize_adapthist` function from `skimage` `exposure` library.
I choose this function because I wanted to enhance contrast of my images. This function uses histograms computed over different tile regions of the image.
I thought about using `keras.preprocessing.image.ImageDataGenerator` to use image rotation, width and height shift and zoom but after training my network on contrast enhanst images I descided to do project submission of my current working algorythm.
#### Model Architecture: The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
To build my model I used skeleton of the network that we build in the classroom. I did a few changes. Instead of excepting grey images I changes it to except color RGB images by changing
`tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6)` to be `tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6)` in the first `Layer` of the network.
The output has changed too, it is 43.
My network has 5 layers. In Layer 1 I create variable for my weights and bias. Run tensor flow `conv2d` function to create convolution, than add bias and call `ft.nn.relu` to activate my layer.
After layer has been activated I do MaxPooling and start my second Layer.
In my second layer I do the same.
Than I my second layer on top of same action from Layer 1 and 2 I add flattening with input = 5x5x16 and output = 400.
I added `dropout` function in my third and fourth layers to prevent my model from over fitting.
After the fifth layer I `return logits`.
Here is a code example of 1Layer 1`
```python
# Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6))
    x = tf.nn.bias_add(x, b1)
    print("layer 1 shape:",x.get_shape())

    # Activation. Bias.
    x = tf.nn.relu(x)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer1 = x
```
#### Model Training: The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.
To train my model I used:
```python
rate = 0.001
BATCH_SIZE = 128
EPOCHS = 30
```
As an optimizer I used `AdamOptimizer`, for cross entropy I did `softmax_cross_entropy_with_logits`, for loss operation I used `reduce_mean` function.
This is a code example of my evaluation function:
```python
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```
Code sample of my code that does training:
```python
print("Training...")
    print()
    for i in range(EPOCHS):
        print("Shuffling data...")
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        print("Evaluating accuracy...")
        validation_accuracy = evaluate(X_train, y_train)
        final_validation_accuracy = validation_accuracy
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("Final Validation Accuracy = {:.3f}".format(final_validation_accuracy))
    print("Model saved")
```

#### Solution Approach: The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.
#### My Final Validation Accuracy = 0.999. My Test Set Accuracy = 0.951

### Test a Model on New Images:
#### Acquiring New Images: The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.
I have downloaded a set of German signs. I used a random set of 16 files type `.ppm`. For better algorithm performance I resided them to be (32x32) pixels.
I tried to select images of different quality and type. Too dark, sign images covered with something, too bright, too blury,..
#### Performance on New Images: The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.
![new images][image11]

#### Model Certainty - Softmax Probabilities: The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.
Here is a sample of code I used to calculate 5 softmax probabilities for the predictions of German sighns
```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.

## This code calculates and missclassification error for each sign and
## show maxximum misclassiffied sign for this particular one
for i in range(n_classes):
    cm = confusion_matrix(y_test, y_pred)
    cm_t = np.transpose(cm)
    total_true_count = np.sum(y_test==i)
    total_true_pred = cm[i][i]
    precision = total_true_pred / np.sum(cm_t[i].ravel())
    cm[i][i] = 0
    max_misclas_as = np.argmax(cm[i])
    percent_clas_max_confusion = cm[i][max_misclas_as] / total_true_count * 100

    print("Class %s:" % label_dict[i])
    print("  Accuracy = {:.2f}%".format(total_true_pred/total_true_count*100))
    print("  Precision = {:.2f}%".format(precision*100))

    if (cm[i][max_misclas_as] != 0):
        print("  Maximum Misclassified as: %s" % label_dict[max_misclas_as])
        print("  Misclassification Percentage for class above: {:.2f}%".format(percent_clas_max_confusion))

    print()
```
Results were:
```
Class Roundabout mandatory:
  Accuracy = 93.33%
  Precision = 98.82%
  Maximum Misclassified as: Speed limit (100km/h)
  Misclassification Percentage for above class: 2.22%

Class End of no passing:
  Accuracy = 93.33%
  Precision = 83.58%
  Maximum Misclassified as: No passing
  Misclassification Percentage for above class: 6.67%
```

You're reading it! and here is a link to my [project code](https://github.com/ooleksyuk/carnd-traffic-sign0classifier-project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 10440
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
![example of data set][image9]

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Data set histogram][image10]

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
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6, VALID padding	|
| Convolution 3x3	    | 1x1 strides, VALID padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16, VALID padding	|
| Fully connected		| Input = 400. Output = 120.					|
| RELU					|												|
| Fully connected		| Input = 84. Output = 43.		    			|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train my model I used:
```python
rate = 0.001
BATCH_SIZE = 128
EPOCHS = 30
```
As an optimizer I used `AdamOptimizer`, for cross entropy I did `softmax_cross_entropy_with_logits`, for loss operation I used `reduce_mean` function.
This is a code example of my evaluation function:
```python
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```
Code sample of my code that does training:
```python
print("Training...")
    print()
    for i in range(EPOCHS):
        print("Shuffling data...")
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        print("Evaluating accuracy...")
        validation_accuracy = evaluate(X_train, y_train)
        final_validation_accuracy = validation_accuracy
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("Final Validation Accuracy = {:.3f}".format(final_validation_accuracy))
    print("Model saved")
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.999
* test set accuracy of 0.951

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Original Image                        | Predicted                             |
|:-------------------------------------:|:-------------------------------------:|
| Dangerous curve to the right          | Dangerous curve to the right          |
| Pedestrians                           | Pedestrians                           |
| Speed limit (120km/h)                 | Speed limit (120km/h)                 |
| Priority road                         | Priority road                         |
| No Passing                            | No Passing                            |
| Speed limit 60(km/h)                  | Speed limit 60(km/h)                  |
| Speed limit 70(km/h)                  | Speed limit 70(km/h)                  |
| Speed limit 30(km/h)                  | Speed limit 30(km/h)                  |
| Road work                             | Road work                             |
| Keep right                            | Keep right                            |
| Right of way at the next intersection | Right of way at the next intersection |
| Speed limit 80(km/h)                  | Speed limit 80(km/h)                  |
| Speed limit 100(km/h)                 | Speed limit 100(km/h)                 |
| Turn left ahead                       | Turn left ahead                       |
| Priority road                         | Priority road                         |
| Speed limit 50(km/h)                  | Speed limit 50(km/h)                  |

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Accuracy         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:| 
| 97.64%         			| Speed limit (30km/h)   									|
| 96.67%    				| Speed limit (20km/h)										|
| 98.53%				| Speed limit (50km/h)											|
| 98.18%      			| No passing for vehicles over 3.5 metric tons				 				|
| 99.42%			    | Priority road     							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

## Traffic Sign Recognition Program

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.
