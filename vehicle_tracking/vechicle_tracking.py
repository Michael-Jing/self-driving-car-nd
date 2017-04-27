from multiprocessing import Pool
import numpy as np
import pandas as pd
from functions import *
import pickle
from moviepy.editor import VideoFileClip


def process_udacity_images():
    """
     I use this function to exctract car and pedestrain images from the images udacity provided and pickle them for
     later use
    :return:
    """
    import pandas as pd
    import pickle
    path = "./object-detection-crowdai"
    data = pd.read_csv(path + "/labels.csv")
    check = data.apply(lambda x: x[0] < x[1] and x[2] < x[3], axis=1)
    data = data[check]
    data.reset_index(inplace=True)
    images = []
    labels = []
    for i in np.arange(data.shape[0]):
        images.append(extract_image(data.loc[i, "xmin"], data.loc[i, "xmax"],
                                    data.loc[i, "ymin"], data.loc[i, "ymax"],
                                    "./object-detection-crowdai/" + data.loc[i, "Frame"], (64, 64)))
        labels.append(get_label(data.loc[i, "Label"]))
        # pe(images))
        # print(images)
    image_array = np.array(images)
    print(image_array.shape)
    image_file = open("udacity_images.p", "wb")
    labels_file = open("udacity_labels.p", "wb")
    pickle.dump(image_array, image_file)
    labels = np.array(labels)
    pickle.dump(labels, labels_file)
    print(labels.shape)


def process_other_data():
    """
    I use this function to process the data provided for this project except those published by udacity.
    :return:

    """
    # filenames for car
    vehicles_gti_far = glob.glob("./vehicles/GTI_Far/*.png")
    vehicles_gti_left = glob.glob("./vehicles/GTI_Left/*.png")
    vehicles_middle_close = glob.glob("./vehicles/GTI_MiddleClose/*.png")
    vehicles_gti_right = glob.glob("./vehicles/GTI_Right/*.png")
    vehicles_kitti = glob.glob("./vehicles/KITTI_extracted/*.png")

    # filenames for non-vehicles
    non_car_gti = glob.glob("./non-vehicles/GTI/*.png")
    non_car_extras = glob.glob("./non-vehicles/Extras/*.png")

    images = []
    from scipy.misc import imread

    for fname in vehicles_gti_far:
        images.append(imread(fname))
    for fname in vehicles_gti_left:
        images.append(imread(fname))
    for fname in vehicles_middle_close:
        images.append(imread(fname))
    for fname in vehicles_gti_right:
        images.append(imread(fname))
    for fname in vehicles_kitti:
        images.append(imread(fname))
    nb_positive = len(images)

    for fname in non_car_gti:
        images.append(imread(fname))
    for fname in non_car_extras:
        images.append(imread(fname))

    labels = np.zeros(len(images))
    labels[np.arange(nb_positive)] = 1
    images = np.array(images)

    image_file = open("course_images.p", "wb")
    labels_file = open("course_labels.p", "wb")
    pickle.dump(images, image_file)

    pickle.dump(labels, labels_file)


clf = joblib.load("./svm_classifier.pkl")
feature_scaler = joblib.load("./feature_scaler.pkl")


def process_image(img):
    detector = CVehicleDetector()
    detector.__init__()
    return detector.detect(img)




def process_video(name):
    file_in = name + ".mp4"
    file_out = name + "_out.mp4"
    clip1 = VideoFileClip(file_in)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(file_out, audio=False)


if __name__ == "__main__":
    # process_udacity_images()
    # process_other_data()
    process_video("project_video")
