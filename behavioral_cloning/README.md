## data preprocessing
    
   -  The top 1/3 of the image is sky most of the time, and the bottom 1/5 is head of the car, so in order to reduce input data size and speed up training, I remove the top 1/3 and the bottom 1/5 of the image,
   -  In order to reduce input data size and speedup training further, I decide to resize image to size 32  $\times$ 64. In my initial tryings, I was using image size 66 $\times $ 200, as stated in the paper End to End Learning for Self-Driving Cars, but when was trying to tune the model architecture, the training process is quite slow, so I decided to reduce the image size to as small as possible.
   -  Normalization of input data is must in order to make the training process easier and more efficient, so I divided pixel values by 255 to scale the values to the range of 0 and 1. 
   -  In order to evaluate the training process and notice overfitting when it happens, I split data into training and validation set.
   -  In my initial tryings, the car drives well on the first track, but hit the border of the road somewhere in the second track, so obviously I need more data to teach the model to recover when the car is about to hit the road border, but it is not easy to gain that kind of data, so I used a trick instead by multiplying the steering angles in training set by 1.2. When the original steering angle is small, it is also small after been multiplied by 1.2, but when the original steering angle is large, it could be much larger after been multiplied by 1.2, so preventing the car hid the road border effectively.

## Data Augmentation

   - The original training data I used is the one provided in the course.

   -   In order to gain as much training data as possible, I   used images from the left and right cameras also and adjust steering angles accordingly. When the image from the left camera is used, add 0.25 to the steering angle, when the image from the right camera is used, minus 0.25 to the steering angle.
   -  In order to teach the model to drive on different weather and lighting conditions, I randomly changed the brightness of the image, as showed in the following image.
    ![random brightness augmentation](https://cdn-images-1.medium.com/max/800/1*LTg_FFgMF1Tgw-dI93lgXw.png)
   -  Mountains, trees, even clouds will project shadows on the ground and thus in the images seen by the car camera, so I added random shadows on the image to teach the model to deal with this effect, as showed in the following image.
   ![random shadows augmentation](https://cdn-images-1.medium.com/max/800/1*I5MyRkjrMc2ohpLL-VxJUA.png)
   **Please be aware that the above three technique including the images showed are from this [blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.tfansn960)**
  
   -   zoom images randomly within the range of 10%

## Model Architecture
    
    The model architecture I used is a simplified version of the one used in the paper End to End Learning for Self-Driving Cars published by researchers in NVIDIA Corporation.
    Frankly speaking, I don't know what would be a general approach to designing neural network architecture except that convolutional neural networks is the common practice in the field of image recognition. So in this task, if I don't read the paper mentioned above, my strategy would be try with a network with a few convolutional lays followed by some number of fully connected layers and then the final output layer, and experiment on parameters like the number of layers, the number of filters, filter size, filter stride, number of neurons in fully connected layers.

My model architecture is summarized as follows:

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

## Model Training

   *  I have issues to use the fit_generator method, so I use fit and get data out of the image generator and feed into fit manually in a while loop,
   *  Most of the training data has steering angle 0 or near 0, so I order to give the model enough training examples for situations where the steering angle is bigger, I use sample_weight parameter in fit method to give samples with bigger steering angles high weight,
   *  I save and then load the model during different training runs, and I used learning rate 0.01 in the beginning and reduced it step by step in later training runs eventually to 0.0001,
   * I only do validation every 20 batches in order to speed up training a little.
