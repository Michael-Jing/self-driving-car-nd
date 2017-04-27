import unittest
from scipy.misc import imread
import cv2
import matplotlib.pyplot as plt
import glob

from functions import *

class CTest(unittest.TestCase):

    def setUp(self):
        img_file = "../CarND-Advanced-Lane-Lines/video_images/imgv231.jpg"
        self.img = imread(img_file)


    def atest_cv2resize(self):
        img_files = glob.glob("./object-detection-crowdai/1479498371963069978.jpg")
        img = imread(img_files[0])
        img = cv2.resize(img, (64, 64))
        plt.imshow(img)
        plt.show()

    def atest_extraction_resize(self):
        img_files = glob.glob("./object-detection-crowdai/1479498371963069978.jpg")
        img = extract_image(100, 300, 100, 400, img_files[0], (64, 64))
        plt.imshow(img)
        plt.show()

    def atest_window_slider(self):
        img_file = "../CarND-Vehicle-Detection/test_images/test1.jpg"
        img = imread(img_file)
        print("image shape is: {}".format(img.shape))
        bottom_half = img[img.shape[0] / 2 : img.shape[0], :]
        window_params = [[64, 0, 128], [96, 32, 224], [128, 48, 256], [160, 64, 288], [224, 80, 320], [320, 40, 360]]
        window_list = window_slider(bottom_half, window_params)
        print(window_list)

    def atest_hog_feature_region(self):
        bottom_half = self.img[self.img.shape[0] / 2: self.img.shape[0], :]
        window_sizes = [64, 128]
        orient = 9
        pix_per_cell = 8
        cell_per_block = 4
        features = hog_for_region(bottom_half, window_sizes, orient, pix_per_cell, cell_per_block, "ALL")
        print("shape for hog feature when window_size is 64")

        print("shape for hog feature when window size is 128")


    def atest_get_hog_selector(self):
        print("test get hog selector")
        window = ((0, 368),(96, 404))
        selector = get_hog_selector(window)
        print(selector)

    def atest_select_hog_for_window(self):
        bottom_half = self.img[self.img.shape[0] / 2: self.img.shape[0], :]
        window_params = [[64, 0, 128], [96, 32, 224], [128, 16, 256], [160, 24, 288], [224, 80, 320], [320, 40, 360]]
        window_list = window_slider(bottom_half, window_params)
        window_sizes = [64, 96, 128, 160, 224, 320]
        orient = 9
        pix_per_cell = 8
        cell_per_block = 4
        hog_array = hog_for_region(bottom_half, window_sizes, orient, pix_per_cell, cell_per_block, channel="ALL")



    def atest_get_spatial_color_feature(self):
        spatial_color_features = get_spatial_color_features(self.img, False, True, 32, 7)
        print("spatial_color features shape")
        print(spatial_color_features.shape)

    def atest_vehicle_detection_pipeline(self):
        plt.imshow(self.img)
        plt.show()
        heat_map = vehicle_detection_pipeline(self.img)
        heat_map[heat_map > 0] = 255
        plt.imshow(heat_map)
        plt.show()

    def atest_window_region(self):
        window_params = [[64, 0, 128], [96, 8, 224], [128, 16, 256], [160, 24, 288], [224, 80, 320], [320, 40, 360]]
        window_params = [[96, 8, 224], [128, 16, 256], [160, 24, 288], [224, 32, 320]]
        for param in window_params:
            window_size = param[0]
            begin = param[1]
            end = param[2]
            step = window_size /  4
            span = end - begin
            num = int((span - window_size) / step + 1)
            draw_img = np.copy(self.img)
            for i in range(num):
                y1 = int(360 + begin + i * step)
                y2 = int(y1 + window_size)
                x1 = i * 50
                x2 = self.img.shape[1]
                color = [1, 1, 1]
                color[i % 3] = 255
                color = tuple(color)
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 6)
            plt.imshow(draw_img)
            plt.show()

    def test_pipeline(self):
        image_files = glob.glob("../CarND-Advanced-Lane-Lines/video_images/imgv6*.jpg")
        detector = CVehicleDetector()
        detector.__init__()
        for fname in image_files:
            image = imread(fname)
            image_copy = np.copy(image)
            result = detector.detect(image)
            result = np.vstack((image_copy, result))
            plt.imshow(result)
            plt.show()


if __name__ == "__main__":
    unittest.main()