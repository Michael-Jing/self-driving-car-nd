
from scipy.misc import imread
import cv2

from sklearn.externals import joblib

import numpy as np
import cv2
from skimage.feature import hog
from scipy.misc import imresize
import pickle
from scipy.ndimage.measurements import label

import glob
import matplotlib.pyplot as plt

def extract_image(xmin, xmax, ymin, ymax, file, size):

    image = imread(file)
    extracted = image[ymin:ymax, xmin:xmax, :]
    return cv2.resize(extracted, size)


def get_label(x):
    if x == "Car" or x == "Truck":
        return 1
    else:
        return 0


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
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


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
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


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
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


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def window_slider(img, params):

    window_list = []
    for param in params:
        window_list += slide_window(img, y_start_stop=[param[1]+360, param[2]+360], xy_window=(param[0], param[0]), xy_overlap=(0.75, 0.75))
    return window_list

def _hog_for_region(img, window_sizes, orient, pix_per_cell, cell_per_block):
    """
    extract hog features for the entire image region, for only one channel, return results as multi dimensional array
    :param img: the bottom half of the img need process
    :return: hog features for different window size.
    """
    hog_features = []
    for window_size in window_sizes:
        scaling_factor = window_size / 64
        scaled_img = imresize(img,  (int(img.shape[0] / scaling_factor), int(img.shape[1] / scaling_factor)))
        features = get_hog_features(scaled_img, orient=orient, pix_per_cell=pix_per_cell,  cell_per_block=cell_per_block,
                                    feature_vec=False)
        hog_features.append(features)
    return hog_features

def hog_for_region(img, window_sizes, orient, pix_per_cell, cell_per_block, channel):
    all_channel_hog_features = []
    if channel == "ALL":
        for i in range(3):
            hog_features = _hog_for_region(img[:,:,i], window_sizes, orient, pix_per_cell, cell_per_block)
            all_channel_hog_features.append(hog_features)
    else:
        hog_features = _hog_for_region(img[:,:,channel], window_sizes, orient, pix_per_cell, cell_per_block)
        all_channel_hog_features.append(hog_features)
    return all_channel_hog_features

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

clf = joblib.load("./svm_classifier.pkl")
feature_scaler = joblib.load("./feature_scaler.pkl")
def vehicle_detection_pipeline(image, feature_scaler, clf):

    heat_map = np.zeros_like(image)
    # get windows
    # get hog feature for the whole region
    # iterator through windows
        # select window region out
        # compute spatial and color features
        # select hog features
        # compose feature vector and predict
        # if positive, update heat map
    # threshold heatmap to remove false positives
    # the find labels part
    # draw back
    bottom_half = image[image.shape[0] / 2: image.shape[0], :, :]
    window_params = [[64, 0, 128], [96, 8, 224], [128, 16, 256], [160, 24, 288], [224, 32, 320], [320, 40, 360]]
    window_sizes = [64, 96, 128, 160, 224, 320]

    # a dict to get window size position in the window_sizes list
    window_size2indices = {}
    for i, j in enumerate(window_sizes):
        window_size2indices[j] = i

    orient = 9
    pix_per_cell = 8
    cell_per_block = 4
    color_space = "YUV"
    hog_channel = "ALL"
    spatial_feat = False
    spatial_size = 32
    hist_feat = True
    hist_bins = 7
    heat_map_threshold = 2

    # apply color conversion
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

    windows = window_slider(bottom_half, window_params)
    whole_hog_features = hog_for_region(feature_image, window_sizes, orient, pix_per_cell, cell_per_block, hog_channel)

    for window in windows:
        # window coordinates format is (x, y)
        x1, y1 = window[0]
        x2, y2 = window[1]
        local_image = feature_image[y1:y2, x1:x2, :]
        window_size = x2 - x1
        if window_size != 64:
            local_image = imresize(local_image, (64, 64))
        spatial_color_features = get_spatial_color_features(local_image, spatial_feat, hist_feat, spatial_size, hist_bins)
        hog_features = []
        hog_indiy1, hog_indiy2, hog_indix1, hog_indix2 = get_hog_selector(window)
        for channel_hog_features in whole_hog_features:
            channel_hog_features_current_window_size = channel_hog_features[window_size2indices.get(window_size)]
            hog_features.append(channel_hog_features_current_window_size[hog_indiy1: hog_indiy1 + 5, hog_indix1: hog_indix1 + 5, :, :, :])
        hog_features = np.ravel(hog_features)
        features = np.concatenate((spatial_color_features, hog_features))
        features = feature_scaler.transform(features)
        # print("feature shape is: {}".format(features.shape))
        result = clf.predict(features)
        # print(result)
        if result > 0:
            heat_map[y1:y2, x1:x2] += 1
    heat_map[heat_map < heat_map_threshold] = 0
    labels = label(heat_map)
    img = draw_labeled_bboxes(image, labels)
    return img












def get_spatial_color_features(image, spatial_feat, hist_feat, spatial_size, hist_bins):
    # Iterate through the list of images
    file_features = []
    feature_image = image

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
            # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)

    return (np.concatenate(file_features))



def get_hog_selector(window):

    top_left = window[0]
    bottom_right = window[1]
    window_size = bottom_right[0] - top_left[0]
    scale_factor = window_size / 64

    y_ind1 = (top_left[1] - 360) / (8 * scale_factor)
    y_ind2 = (bottom_right[1] - 360) / (8 * scale_factor)
    x_ind1 = top_left[0] / (8 * scale_factor)
    x_ind2 = bottom_right[0] / (8 * scale_factor)
    if y_ind1 < 0:
        print(window)
    return int(y_ind1), int(y_ind2), int(x_ind1), int(x_ind2)


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
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
    nx_windows = np.int((xspan - xy_window[0])/ nx_pix_per_step) + 1
    ny_windows = np.int((yspan - xy_window[1])/ ny_pix_per_step) + 1
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


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows

class CVehicleDetector(object):

    def __init__(self):
        # define all parameters need to be pre defined
        # keep heat map
        self.clf = joblib.load("./svm_classifier.pkl")
        self.feature_scaler = joblib.load("./feature_scaler.pkl")
        self.heat_map_decay_factor = 0.9
        self.heat_map = np.zeros((720, 1280))

        self.window_params = [[64, 0, 128], [96, 8, 224], [128, 16, 256], [160, 24, 288], [224, 32, 320], [320, 40, 360]]
        self.window_sizes = [64, 96, 128, 160, 224, 320]

        # a dict to get window size position in the window_sizes list
        self.window_size2indices = {}
        for i, j in enumerate(self.window_sizes):
            self.window_size2indices[j] = i

        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 4
        self.color_space = "YUV"
        self.hog_channel = "ALL"
        self.spatial_feat = False
        self.spatial_size = 32
        self.hist_feat = True
        self.hist_bins = 7
        self.heat_map_threshold = 2
        self.windows = window_slider(np.zeros((360, 1280)), self.window_params)



    def detect(self, image):
        if self.color_space != 'RGB':
            if self.color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif self.color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif self.color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif self.color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif self.color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        whole_hog_features = hog_for_region(feature_image, self.window_sizes, self.orient, self.pix_per_cell, self.cell_per_block,
                                            self.hog_channel)
        for window in self.windows:
            # window coordinates format is (x, y)
            x1, y1 = window[0]
            x2, y2 = window[1]
            local_image = feature_image[y1:y2, x1:x2, :]
            window_size = x2 - x1
            if window_size != 64:
                local_image = imresize(local_image, (64, 64))
            spatial_color_features = get_spatial_color_features(local_image, self.spatial_feat, self.hist_feat, self.spatial_size,
                                                                self.hist_bins)
            hog_features = []
            hog_indiy1, hog_indiy2, hog_indix1, hog_indix2 = get_hog_selector(window)
            for channel_hog_features in whole_hog_features:
                channel_hog_features_current_window_size = channel_hog_features[self.window_size2indices.get(window_size)]
                hog_features.append(
                    channel_hog_features_current_window_size[hog_indiy1: hog_indiy1 + 5, hog_indix1: hog_indix1 + 5, :,
                    :, :])
            hog_features = np.ravel(hog_features)
            features = np.concatenate((spatial_color_features, hog_features))
            features = feature_scaler.transform(features)
            # print("feature shape is: {}".format(features.shape))
            result = clf.predict(features)
            # print(result)
            if result > 0:
                self.heat_map[y1:y2, x1:x2] += 1

        self.heat_map *= self.heat_map_decay_factor
        self.heat_map[self.heat_map < self.heat_map_threshold] = 0
        labels = label(self.heat_map)
        img = draw_labeled_bboxes(image, labels)
        return img



