# **Behavioral Cloning**


---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md  summarizing the results

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.json
```

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 98-106)

The model includes RELU layers to introduce nonlinearity (code line 103, 105, 107, 110, 113)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 111).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 167, 182). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116).

#### 4. Appropriate training data
I used the training data provided in the course, and in order to train the model to recover from side of the road, I multiplied the steering angle in training data by 1.2, which has small effect when the car drives in the center of the road while a bigger effect when the car need a recover.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was try an error.

My first step was training to use transfer learning and do feature extraction on the vgg16 model because I think it's not too complex and are one of the state of art network design for image recognition tasks.

My feature extraction on vgg16 was not successful and I turn to the paper End to End Learning for Self-Driving Cars published by researchers in NVIDIA. I was surprised that their network design is quite simple, and realized that my implementation should be even simpler, after all, my model only need to drive the car on the simulated track which is much simpler than real roads situations. So I used a simplified version of the neural network used in the paper.

 I guess I did well on training data augmentation, and feed the model more than enough data to train, I did not notice obvious overfitting, but I added a dropout layer anyway.


Then  I run the simulator to see how well the car was driving around track one. and it was quite well, so I tried on track two(the old mountain theme), and the car was unable to make some turns. To overcome this problem, I multiplied the steering angle in the training set by 1.2, and also set higher sample weight for training samples with higher steering angle.

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road. One issue is that the car was driving a little zig-zag around track one, so I think the model was a little overfitted to the amplified steering angle in the training data, but I did not notice zig-zag behavior when the car drives around track two.

#### 2. Final Model Architecture

The final model architecture (model.py lines 98-114) consisted of a convolution neural network with the following layers and layer sizes


-    input size is 32 by 64 by 3
-   convolutional layer with 24 3 by 3 filters， stride 1 by 1
    (since I reduced the size of the input image dramatically, I think the convolutional layer should now look at a smaller region, so I changed the 5 by 5 filters in the paper to 3 by 3 filters)
- convolutional layer with 36 3 by 3 filters, stride 1 by 1, activation relu
-  convolutional layer with 48 3 by 3 filters, stride 2 by 2, activation relu
-  convolutional layer with 64 3 by 3 filters, stride 2 by 2, activation relu
-  flatten layer
-  dense layer with 100 neurons, activation relu
-  dropout layer with 25% possibility to drop a neuron
(dropout layer is added to help prevent overfitting)
-  dense layer with 10 neurons, activation relu
-  the final output layer, with only 1 neuron.


#### 3. Creation of the Training Set & Training Process

I used the training data provided in the course to save time.

To augment the data set
- I used camera pictures from the left and right cameras also, and adjusted the steering angel by 0.25.
- I randomly changed the brightness of the image as showing below
![random brightness augmentation](https://cdn-images-1.medium.com/max/800/1*LTg_FFgMF1Tgw-dI93lgXw.png)
- I added random shadows to the image as showing below
![random shadows augmentation](https://cdn-images-1.medium.com/max/800/1*I5MyRkjrMc2ohpLL-VxJUA.png)

**Please be aware that the above three technique including the images showed here are from this [blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.tfansn960)**


- I randomly zoomed the images within the range of 10%

I then preprocessed this data by
- remove the top 1/3 and bottom 1/5 of the image, to speedup training,
- resize the image to size 32 $\times$ 64, to speedup training,
- rescale the pixel values to the range 0 and 1


- I finally randomly shuffled the data set and put 33% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used generator to feed data to the training process, so number of epochs wasn't necessary.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
