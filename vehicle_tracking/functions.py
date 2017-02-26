from scipy.misc import imread
import numba
import random
from sklearn.externals import joblib

import numpy as np
import cv2
from skimage.feature import hog
from scipy.misc import imresize
from scipy.ndimage.measurements import label
import multiprocessing
from multiprocessing import Pool


def extract_image(xmin, xmax, ymin, ymax, file, size):
    """
    This function is used to extract images from the udacity published data,
    :param xmin: defines the top left corner of the interested region together with ymin
    :param xmax: defines the bottom right corner of the interested region together with ymax
    :param ymin:
    :param ymax:
    :param file: image file
    :param size: desired image size
    :return: image extracted with size defined by size param
    """
    image = imread(file)
    extracted = image[ymin:ymax, xmin:xmax, :]
    return cv2.resize(extracted, size)


def get_label(x):
    """
    convert String labels in the udacity published data to numeric labels
    :param x: String labels
    :return: numeric labels
    """
    if x == "Car" or x == "Truck":
        return 1
    else:
        return 0


def augment_rescale(img):
    """
    zoom image by 10 to 30 percent randomly.
    :param img:
    :return:
    """
    y, x, z = img.shape
    scale_ratio = random.uniform(1.1, 1.3)
    zoomed = cv2.resize(img, (int(scale_ratio * y), int(scale_ratio * x)))
    return zoomed[:y, :x, :]


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


def get_hog_features(image, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    extract hog features from image
    :param image: image to extract features from
    :param orient: number of orientations
    :param pix_per_cell:
    :param cell_per_block:
    :param vis:
    :param feature_vec:
    :return:
    """
    # Call with two outputs if vis==True
    img = np.copy(image)
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    """
    get spatially binned image
    :param img:
    :param size:
    :return:
    """
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features



def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    compute color histogram features
    :param img:
    :param nbins:
    :param bins_range:
    :return:
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_features_from_array(imgs, color_space='RGB', spatial_size=(32, 32),
                                hist_bins=32, orient=9,
                                pix_per_cell=8, cell_per_block=2, hog_channel=0,
                                spatial_feat=True, hist_feat=True, hog_feat=True):

    """
    function to extract spatial, color histogram and hog features from a 4 D array with shape [num_images, height, width, depth]
    :param imgs:
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for row in np.arange(imgs.shape[0]):
        file_features = []
        # Read in each one by one
        image = imgs[row]
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return np.array(features)


def extract_feature_one_img(image, spatial_size=(32, 32),
                            hist_bins=32, orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    extract spatial, color histogram and hog features from a single image with shape [height, width, depth]
    :param image:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """
    # Create a list to append feature vectors to
    # Iterate through the list of images
    file_features = []
    # Read in each one by one


    feature_image = np.copy(image)

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        file_features.append(hog_features)
    # Return list of feature vectors
    feature_num = len(file_features)
    if feature_num == 1:
        return file_features[0]
    elif feature_num == 2:
        return np.concatenate((file_features[0], file_features[1]))
    else:
        return np.concatenate((file_features[0], file_features[1], file_features[2]))


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    extract spatial, color histogram and hog features from imgaes
    :param imgs: collection of image paths
    :param color_space:
    :param spatial_size:
    :param hist_bins:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param hog_channel:
    :param spatial_feat:
    :param hist_feat:
    :param hog_feat:
    :return:
    """
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def extract_hog_features_one_img(image, orient=9,
                                 pix_per_cell=8, cell_per_block=2, hog_channel=0):

    """extract hog features from a sigle image"""
    feature_image = np.copy(image)

    # Call get_hog_features() with vis=False, feature_vec=True
    hog_features = []
    if hog_channel == 'ALL':
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                 orient, pix_per_cell, cell_per_block,
                                                 vis=False, feature_vec=False))

    else:
        hog_features.append(get_hog_features(feature_image[:, :, hog_channel], orient,
                                             pix_per_cell, cell_per_block, vis=False, feature_vec=False))
    # Append the new feature vector to the features list
    return hog_features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def window_slider(img, params):
    """
    find all window positions in img giving parameters defined by params
    :param img: img to find windows
    :param params: a list defining parameters for different windows to find in the form [[size1, y_start_position,
    y_end_position], [size2, y_start_position, y_end_position], ...]
    :return: a list of tuples in the form ((x1, y1), (x2, y2))
    """
    window_list = []
    for param in params:
        window_list += slide_window(img, y_start_stop=[param[1], param[2]], xy_window=(param[0], param[0]),
                                    xy_overlap=(0.75, 0.75))
    return window_list



def _hog_for_region(img, window_sizes, orient, pix_per_cell, cell_per_block):
    """
    extract hog features for the entire image region, for only one channel, image rescaling is used in order to make
    select features for windows with different sizes easy
    :param img: the bottom half of the img need process
    :return: a list of hog features, each entry corresponds to a different window size, [[hog feature for window size a],
     [hog feature for window size b], ...]
    """
    hog_features = []
    for window_size in window_sizes:
        scaling_factor = 64 / window_size
        scaled_img = imresize(img, scaling_factor)
        features = get_hog_features(scaled_img, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                    feature_vec=False)
        hog_features.append(features)
    return hog_features


def hog_for_region(img, window_sizes, orient, pix_per_cell, cell_per_block, channel):
    """
    extract hog features for the entire image region, for the channel specifed or all channels.
    :param img:
    :param window_sizes:
    :param orient:
    :param pix_per_cell:
    :param cell_per_block:
    :param channel:
    :return: a list of list containing hog features, in the form [[hog feature for channel 1], [hog feature for channel 2],
     [hog feature for channel 3] ]
    """
    all_channel_hog_features = []
    if channel == "ALL":
        for i in range(3):
            hog_features = _hog_for_region(img[:, :, i], window_sizes, orient, pix_per_cell, cell_per_block)
            all_channel_hog_features.append(hog_features)
    else:
        hog_features = _hog_for_region(img[:, :, channel], window_sizes, orient, pix_per_cell, cell_per_block)
        all_channel_hog_features.append(hog_features)
    return all_channel_hog_features


def draw_labeled_bboxes(img, labels):
    """
    draw rectangle around detected cars
    :param img:
    :param labels:
    :return:
    """
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img




def get_spatial_color_features(image, spatial_feat, hist_feat, spatial_size, hist_bins):
    """
    get spatial and color histogram features from an image.
    :param image:
    :param spatial_feat:
    :param hist_feat:
    :param spatial_size:
    :param hist_bins:
    :return:
    """
    # Iterate through the list of images
    file_features = []
    feature_image = image

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=(spatial_size, spatial_size))
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)

    return (np.concatenate(file_features))


def get_hog_selector(window, pix_per_cell, cell_per_block):
    """
    compute indexes to select hog features from hog features of the entire image
    :param window:
    :param pix_per_cell:
    :return:
    """
    (x1, y1), (x2, y2) = window
    window_size = x2 - x1
    scale_factor = window_size / 64

    y_ind1 = y1 / (pix_per_cell * scale_factor)
    y_ind2 = y2 / (pix_per_cell * scale_factor) - cell_per_block + 1
    x_ind1 = x1 / (pix_per_cell * scale_factor)
    x_ind2 = x2 / (pix_per_cell * scale_factor) - cell_per_block + 1
    if y_ind1 < 0:
        print(window)
    return int(y_ind1), int(y_ind2), int(x_ind1), int(x_ind2)


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    compute window coordinates giving window size and search region
    :param img:
    :param x_start_stop:
    :param y_start_stop:
    :param xy_window:
    :param xy_overlap:
    :return:
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int((xspan - xy_window[0]) / nx_pix_per_step) + 1
    ny_windows = np.int((yspan - xy_window[1]) / ny_pix_per_step) + 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """draw a list of boxes on img"""
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def singleton(cls, *args, **kw):
    """
    singleton class decorator
    :param cls:
    :param args:
    :param kw:
    :return:
    """
    instances = {}

    def _singleton():
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class CVehicleDetector(object):
    """
    Detect vehicles
    """
    def __init__(self):
        # load pretrained models
        self.clf = joblib.load("./svm_classifier2.pkl")
        self.feature_scaler = joblib.load("./feature_scaler2.pkl")

        # keep a heat map over frames
        self.heat_map_decay_factor = 0.8
        self.heat_map = np.zeros((720, 1280))

        # define parametrs
        self.window_params = [ [96, 8, 196], [112, 8, 196], [136, 8, 256], [160, 16, 288], [192, 32, 320], [224, 48, 320]]
        self.window_sizes = [96, 112, 136, 160, 192, 224]
        self.orient = 9
        self.pix_per_cell = 16
        self.cell_per_block = 2
        self.color_space = "YUV"
        self.hog_channel = "ALL"
        self.spatial_feat = False
        self.spatial_size = 32
        self.hist_feat = True
        self.hist_bins = 32
        self.heat_map_threshold = 10
        self.positive_windows = None
        self.image = None
        # compute window list
        self.windows = window_slider(np.zeros((360, 1280)), self.window_params)

        # a dict to get window size position in the window_sizes list
        self.window_size2indices = {}
        for i, j in enumerate(self.window_sizes):
            self.window_size2indices[j] = i

    def feed_image(self, image):
        """
        detect cars in image
        :param image:
        :return: an image with boxes drawing around cars.
        """
        self.image = image
        # select bottom half out
        bottom_half = image[image.shape[0] / 2: image.shape[0], :, :]
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(bottom_half)

        # hog features for the whole bottom half
        whole_hog_features = hog_for_region(feature_image, self.window_sizes, self.orient, self.pix_per_cell,
                                            self.cell_per_block,
                                            self.hog_channel)
        all_features = []
        for window in self.windows:
            # window coordinates format is (x, y)
            (x1, y1), (x2, y2) = window

            local_image = feature_image[y1:y2, x1:x2, :]
            window_size = x2 - x1
            if window_size != 64:
                local_image = imresize(local_image, (64, 64))
            # compute spatial and color histogram features since hog features will be selected other than computed
            spatial_color_features = get_spatial_color_features(local_image, self.spatial_feat, self.hist_feat,
                                                                self.spatial_size,
                                                                self.hist_bins)
            hog_features = []
            # indices to select hog features from whole_hog_features
            hog_indiy1, hog_indiy2, hog_indix1, hog_indix2 = get_hog_selector(window, self.pix_per_cell, self.cell_per_block)

            for channel_hog_features in whole_hog_features:
                # hog features for the current window size
                channel_hog_features_current_window_size = channel_hog_features[
                    self.window_size2indices.get(window_size)]
                # hog features for current window
                hog_features_current_window = channel_hog_features_current_window_size[
                                              hog_indiy1: hog_indiy2,
                                              hog_indix1: hog_indix2, :, :, :]
                hog_features.append(hog_features_current_window)
            if len(hog_features) == 1:
                hog_features = np.array(hog_features)
            elif len(hog_features) == 2:
                hog_features = np.concatenate((hog_features[0], hog_features[1]))
            else:
                hog_features = np.concatenate((hog_features[0], hog_features[1], hog_features[2]))

            hog_features = np.ravel(hog_features)

            features = np.concatenate((spatial_color_features, hog_features))

            all_features.append(features)

        # features for all windows in the image
        all_features = np.array(all_features)
        # scale and predict
        all_features_scaled = self.feature_scaler.transform(all_features)
        result = self.clf.predict(all_features_scaled)

        # decay the effect of old heat_map

        # find positive predictions and update heat_map
        self.positive_windows = result.nonzero()
        self.heat_map *= self.heat_map_decay_factor

    def get_detected_cars(self):

        for i in np.arange(self.positive_windows[0].shape[0]):
            window_index = self.positive_windows[0][i]
            (x1, y1), (x2, y2) = self.windows[window_index]
            self.heat_map[y1 + 360:y2 + 360, x1:x2] += 1

        # heat_map thresholding
        self.heat_map[self.heat_map < self.heat_map_threshold] = 0

        # find labels and draw boxes
        labels = label(self.heat_map)
        img = draw_labeled_bboxes(self.image, labels)
        return img

    def get_boxes_img(self):
        """
        draw individual boxes on image
        :param image:
        :return: an image with boxes drawing around cars.
        """
        img = np.copy(self.image)
        for i in np.arange(self.positive_windows[0].shape[0]):
            window_index = self.positive_windows[0][i]
            (x1, y1), (x2, y2) = self.windows[window_index]
            self.heat_map[y1 + 360:y2 + 360, x1:x2] += 1
            cv2.rectangle(img, (x1, y1 + 360), (x2, y2 + 360), (0, 0, 255), 6)

        return img, self.heat_map

    def get_labels_map(self):
        for i in np.arange(self.positive_windows[0].shape[0]):
            window_index = self.positive_windows[0][i]
            (x1, y1), (x2, y2) = self.windows[window_index]
            self.heat_map[y1 + 360:y2 + 360, x1:x2] += 1

        # heat_map thresholding
        self.heat_map[self.heat_map < self.heat_map_threshold] = 0

        # find labels and draw boxes
        labels = label(self.heat_map)
        return labels[0]


### these global parameters are defined in order to trying have a multithreadnig pipeline
feature_img = np.zeros((360, 1280))
whole_hog_features = None
window_size2indices = {}
orient = 9
pix_per_cell = 16
cell_per_block = 2


spatial_feat = False
spatial_size = 32
hist_feat = True
hist_bins = 32
hog_channel = "ALL"
window_params = [ [96, 8, 196], [112, 8, 196], [136, 8, 256], [160, 16, 288], [192, 32, 320], [224, 48, 320]]
windows = window_slider(np.zeros((360, 1280)), window_params)

cpu_count = multiprocessing.cpu_count()


@singleton
class CVehicleDetector2(object):
    """
    almost the same as CVehicleDetector, but I was trying to use parallel computing to speedup processing.
    """
    def __init__(self):
        # define all parameters need to be pre defined
        # keep heat map
        self.clf = joblib.load("./svm_classifier2.pkl")
        self.feature_scaler = joblib.load("./feature_scaler2.pkl")
       # self.pca = joblib.load("pca2.pkl")
        self.heat_map_decay_factor = 0.9
        self.heat_map = np.zeros((720, 1280))

        self.window_params = [ [96, 8, 196], [112, 8, 196], [136, 8, 256], [160, 16, 288], [192, 32, 320], [224, 48, 320]]
        self.window_sizes = [96, 112, 136, 160, 192, 224]
        # self.window_params = [[64, 0, 360]]
        # self.window_sizes = [64]

        # a dict to get window size position in the window_sizes list


        for i, j in enumerate(self.window_sizes):
            window_size2indices[j] = i

        self.heat_map_threshold = 10
        self.color_space = "YUV"




    def detect(self, image):
        bottom_half = image[image.shape[0] / 2: image.shape[0], :, :]
        global feature_image
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(bottom_half, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(bottom_half)

        global whole_hog_features
        whole_hog_features = hog_for_region(feature_image, self.window_sizes, orient, pix_per_cell,
                                            cell_per_block,
                                            hog_channel)

        all_features = []

        p = Pool()
        for i in range(cpu_count):
            result = p.apply_async(get_features, args=(i, ))
            all_features.append(result)
        p.close()
        p.join()
        features = []
        for result in all_features:
            features += result.get()

        features_array = np.array(features)

        all_features_scaled = self.feature_scaler.transform(features_array)
        # all_features_scaled_pc = self.pca.transform(all_features_scaled)
        # print("feature shape is: {}".format(features.shape))
        result = self.clf.predict(all_features_scaled)
        positive_windows = result.nonzero()
        for i in np.arange(positive_windows[0].shape[0]):
            window_index = positive_windows[0][i]
            (x1, y1), (x2, y2) = windows[window_index]
            self.heat_map[y1 + 360:y2 + 360, x1:x2] += 1

        self.heat_map *= self.heat_map_decay_factor
        self.heat_map[self.heat_map < self.heat_map_threshold] = 0
        # nonzeroy, nonzerox = self.heat_map.nonzero()
        # for i in range(nonzeroy.shape[0]):
        #    image[nonzeroy[i], nonzerox[i]] = 255
        labels = label(self.heat_map)
        img = draw_labeled_bboxes(image, labels)
        return img


def get_features(pid):
    """
    compute features for a window in parallel
    :param pid: a number to distinguish different threads
    :return: features for all windows in an image.
    """
    step = int(len(windows) / cpu_count)

    local_windows = windows[pid * step : (pid + 1) * step]
    if pid == cpu_count - 1:
        local_windows = windows[pid * step:]

    all_features = []
    for window in local_windows:
        # window coordinates format is (x, y)
        (x1, y1), (x2, y2) = window

        local_image = feature_image[y1:y2, x1:x2, :]
        window_size = x2 - x1
        if window_size != 64:
            local_image = imresize(local_image, (64, 64))
        spatial_color_features = get_spatial_color_features(local_image, spatial_feat, hist_feat,
                                                            spatial_size,
                                                            hist_bins)
        hog_features = []
        hog_indiy1, hog_indiy2, hog_indix1, hog_indix2 = get_hog_selector(window, pix_per_cell)
        for channel_hog_features in whole_hog_features:
            channel_hog_features_current_window_size = channel_hog_features[
                window_size2indices.get(window_size)]

            hog_features_current_window = channel_hog_features_current_window_size[
                                          hog_indiy1: hog_indiy2 - cell_per_block + 1,
                                          hog_indix1: hog_indix2 - cell_per_block + 1, :, :, :]
            hog_features.append(hog_features_current_window)
        if len(hog_features) == 1:
            hog_features = np.array(hog_features)
        elif len(hog_features) == 2:
            hog_features = np.concatenate((hog_features[0], hog_features[1]))
        else:
            hog_features = np.concatenate((hog_features[0], hog_features[1], hog_features[2]))

        hog_features = np.ravel(hog_features)
        features = np.concatenate((spatial_color_features, hog_features))

        all_features.append(features)
        return all_features
