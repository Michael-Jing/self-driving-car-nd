
# coding: utf-8
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os.path
from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
from scipy.misc import imread
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential

# load in image data and driving angle data
# split data into training and validation
# define network architecture
# train the model
# test on the simulator


# Help functions
def image_compression(image):
    """cut off the top 1/3 and the bottom 1/5 of the image and then
    resize it to shape(32, 64)"""
    height, width, depth = image.shape
    region = image[int(height / 3) : int(height * 4 / 5)]
    return imresize(region, (32, 64))

# this function is from  https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jqnd5bnzd
def augment_brightness(image):
    """change brightness of the image to simulator different weather
    conditions, this function is from the web"""
    image = image.astype(np.uint8)
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

# this function is from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jqnd5bnzd
def add_random_shadow(image):
    """randomly choose a region and add shadow to it, this function is
    from the web"""
    image = image.astype(np.uint8)
    top_y = 32*np.random.uniform()
    top_x = 0
    bot_x = 16
    bot_y = 32*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def normalize(image):
    """rescale the data to betwwen 0 nad 1"""
    return image / 255

def get_training_data():
    """read in training data, adjust steering angle for left and right
    camera images, return data in numpy array format"""
    log = pd.read_csv("data/driving_log.csv")
    center = log[["center", "steering"]]
    left = log[["left", "steering"]]
    left["steering"] = left["steering"] + 0.25
    right = log[["right", "steering"]]
    right["steering"] = right["steering"] - 0.25
    right.columns = ["center", "steering"]
    left.columns = ["center", "steering"]
    log_data = pd.concat([center, left, right])
    y_train = np.array(log_data["steering"])
    X_train = log_data["center"].apply(lambda x: image_compression(imread("data/"+x.strip())))
    X_train = list(X_train)
    X_train = np.array(X_train)

    return X_train, y_train




def construct_model():
    input_shape = (32, 64, 3)
    model = Sequential()
    model.add(Convolution2D(24, 3, 3,
                            border_mode='valid',
                            input_shape=input_shape))

    model.add(Convolution2D(36, 3, 3, border_mode="valid"))
    model.add(Activation("relu"))
    model.add(Convolution2D(48, 3, 3, border_mode="valid", subsample=(2, 2)))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(2, 2)))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Activation("relu"))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(nb_batch, model, file_name, train_data_generator, validation_data, BATCH_SIZE):
    batch_counter = 0
    model_file = file_name + ".h5"
    if os.path.exists(model_file):
        print("model loaded")
        model = load_model(model_file)
    while batch_counter < nb_batch:
        train_data, train_label = next(train_data_generator)

        if batch_counter % 20 == 0:
            model.fit(train_data, train_label, BATCH_SIZE * 100, nb_epoch = 1, verbose = 2,
                      validation_data=validation_data, sample_weight = np.absolute(train_label) + 0.1)
            model.save(model_file)
            print("model saved")

        else:
            model.fit(train_data, train_label, BATCH_SIZE * 100, nb_epoch = 5, verbose=0,
                      sample_weight = np.absolute(train_label) + 0.1)

        batch_counter += 1

    print("training done")

## save architecture to json
def save_model_to_json(file_name):
    model_file = file_name + ".h5"
    model_archi_file = file_name + ".json"
    model = load_model(model_file)
    json_string = model.to_json()

    with open(model_archi_file, "w") as text_file:
        text_file.write(json_string)

def image_process(batch):
    while True:
        x, y = next(batch)
        num_sample = x.shape[0]
        for i in range(num_sample):
            image = x[i]
            brightness_augmented = augment_brightness(image)
            random_shadow_added = add_random_shadow(brightness_augmented)
            x[i] = normalize(random_shadow_added)
        yield x, y

def main():
    BATCH_SIZE = 128
    # get data
    X, y = get_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 0)
    y_train = y_train * 1.2 # this helps a lot to get the car back when it is off the center of the road
    # after using this technique, the model works well on the second track, but the car goes a little
    # zig-zag on the first track"
    X_test = X_test.astype(np.float64)
    X_test = X_test / 255 # normalize validation data, training data is normalized in data generator



    datagen = ImageDataGenerator(
            zoom_range=0.1,
            fill_mode='nearest')
    batch_gen = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    batch2 = image_process(batch_gen)
    model = construct_model()
    train_model(5000, model, "test", batch2, (X_test, y_test), BATCH_SIZE)
    save_model_to_json("test")

if __name__ == "__main__":
    main()

