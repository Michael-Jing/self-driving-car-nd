

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700

import glob
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import cv2
import logging
logging.basicConfig(filename='lane_line.log', level=logging.DEBUG)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)

        plt.imshow(img)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720),None,None)

distorted_img = imread("camera_cal/calibration10.jpg")
undistored_img = cv2.undistort(distorted_img, mtx, dist, None, mtx)
camera_calibration_img = np.hstack((distorted_img, undistored_img))
imsave("writeup_images/camera_calibration_img.jpg", camera_calibration_img)

def get_transform_matrix(img):
    img_size = (img.shape[1], img.shape[0])

    print("get_transform_matrix, img size is {}".format(img_size))
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])


    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv

M, Minv = get_transform_matrix(undistored_img)

def transform(img, M):
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

origin_img = imread("./writeup_images/imgv400.jpg")
undisted = cv2.undistort(origin_img, mtx, dist, None, mtx)
transformed_img = transform(undisted, M)
img_size = origin_img.shape[1], origin_img.shape[0]
src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])

# print(src.shape)
cv2.polylines(undisted, np.int_([src]), True, (255, 0, 0), 5)
cv2.polylines(transformed_img, np.int_([dst]), True, (255, 0, 0), 5)
imsave("writeup_images/transform.jpg", np.hstack((undisted, transformed_img)))


# function to filter out yellow and white pixels in a image
def filter_yellow_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # mask for yellow color
    lower = np.array([15,70,100])
    upper = np.array([45,255,255])
    mask1 = cv2.inRange(hsv, lower, upper)
    
    # mask for white color
    lower_w = np.array([0, 0, 170])
    upper_w = np.array([255, 23, 255])
    mask2 = cv2.inRange(hsv, lower_w, upper_w)
    mask = cv2.bitwise_or(mask2, mask1)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    return res

# filter out the pixels with the highest value
def most_bright(v_channel):
    indi = v_channel >= np.percentile(v_channel, q=99)
    filtered_v = np.zeros_like(v_channel)
    filtered_v[indi] = 1
    return filtered_v


def sobelx(l_channel, sx_thresh=(20, 100), ksize=5):
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=ksize)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    return sxbinary


def detect_edge_v2(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # range for yellow color
    lower = np.array([15, 50, 100])
    upper = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv_img, lower, upper)
    v_channel = hsv_img[:, :, 2]
    s_channel = hsv_img[:, :, 1]

    v_channel_edge = sobelx(v_channel, sx_thresh=(20, 100), ksize=5)
    s_channel_edge = sobelx(s_channel)
    yellow_mask = cv2.bitwise_and(yellow_mask, s_channel_edge)
    # v_channel = cv2.GaussianBlur(v_channel,(9,9),0)

    v_channel_most_bright = most_bright(v_channel)
    v_channel_most_bright = cv2.blur(v_channel_most_bright * 255, (9, 9))
    v_channel_most_bright[v_channel_most_bright > 0] = 1

    v_edge_most_bright = cv2.bitwise_and(v_channel_edge, v_channel_most_bright)
    # s_channel = hsv_img[:,:,1]
    # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(10,10))
    # cl1 = clahe.apply(s_channel)
    # s_channel_edge = sobelx(s_channel, (20, 100), ksize=5)
    edge = cv2.bitwise_or(v_edge_most_bright, yellow_mask)
    edge[edge > 0] = 1

    return edge

img = imread("./writeup_images/img0.jpg")
binary_img = detect_edge_v2(img)
binary_img = np.dstack((binary_img * 255, binary_img * 255, binary_img * 255))
binary_img_to_show = np.hstack((img, binary_img))
imsave("./writeup_images/binary_image.jpg", binary_img_to_show)


def gradient_thresholding(img, ksize):
    sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = ksize))
    sobely = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = ksize))
    scaled_sobelx = np.uint8(255 * sobelx / np.max(sobelx))
    scaled_sobely = np.uint8(255 * sobely / np.max(sobely))




def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if orient == 'x':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    elif orient == 'y':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    else:
        print("orient should be eighter 'x' or 'y'")
        return None
    # Apply threshold
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    grad_binary = np.zeros_like(gray)
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    scaled_sobelx = np.uint8(255 * sobelx / np.max(sobelx))
    scaled_sobely = np.uint8(255 * sobely / np.max(sobely))
    sobel_mag = np.sqrt(scaled_sobelx ** 2 + scaled_sobely ** 2)
    sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    mag_binary = np.zeros_like(sobel_mag)
    mag_binary[(sobel_mag > mag_thresh[0]) & (sobel_mag < mag_thresh[1])] = 1
    # Calculate gradient magnitude
    # Apply threshold
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    dri = np.arctan2(sobely, sobelx)
    # Calculate gradient direction
    dir_binary = np.zeros_like(dri)
    dir_binary[(dri > thresh[0]) & (dri < thresh[1])] = 1
    # Apply threshold
    return dir_binary




# Edit this function to create your own pipeline.
def detect_edge(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary



def draw_points(img, ally, allx, color = (255, 255, 255)):
    p_left, p_right = np.vstack((ally[0], allx[0])).T, np.vstack((ally[1], allx[1])).T

    p_left = p_left.astype(np.int32)
    p_right = p_right.astype(np.int32)
    number = p_left.shape[0]
    for i in np.arange(number):
        img[p_left[i][0], p_left[i][1]] = np.array(color)
    for i in np.arange(p_right.shape[0]):
        img[p_right[i][0], p_right[i][1]] = np.array(color)
    return img.astype(np.uint8)




# In[72]:

class CPointsDetector(object):
    
    def __init__(self, img, peak, nwindows):
        self.img = img
        self.nonzero = img.nonzero()
        self.nonzeroy = self.nonzero[0]
        self.nonzerox = self.nonzero[1]
        self.nwindows = nwindows
        self.window_height = np.int(self.img.shape[0] / nwindows)
        self.peak = peak
        self.search_width = 25
        self.CENTER_SHIFT = 75
        self.center_shift = self.CENTER_SHIFT
        self.CENTER_SHIFT_STEP = 8
        
    def get_points(self):
        self.center_shift = self.CENTER_SHIFT
        peak = self.peak
        return self.find_points(self.img, peak)
        
    def find_points_in_band_one_center(self, band, rows, center):
        width = band.shape[1]
        left = center - self.search_width
        right = center + self.search_width
        indices = (self.nonzerox >= left) & (self.nonzerox <= right) & (self.nonzeroy >= rows[0]) & (self.nonzeroy <= rows[1])
        
        return self.nonzeroy[indices], self.nonzerox[indices]
                
        
    
    

    def find_points_in_band(self, band, rows, center):
        width = band.shape[1]
        points = np.array([]), np.array([])
        for c in np.arange(max(0, center - self.center_shift), min(width - 1, center + self.center_shift), self.CENTER_SHIFT):
            ally, allx = self.find_points_in_band_one_center(band, rows, c)
            if ally.shape[0] > points[0].shape[0] and ally.shape[0] > 0.5 * (rows[1] - rows[0]):
                points = ally, allx
                self.center_shift = self.CENTER_SHIFT
        if points[0].shape[0] == 0:
            self.center_shift += self.CENTER_SHIFT_STEP
        return points


    def find_points(self, image, peak):
        height = image.shape[0]
        all_points = np.array([]), np.array([])
        center = peak
        for window in range(self.nwindows):
            points = self.find_points_in_band(image, 
                        (max(0, height - (window + 1) * self.window_height),
                         height - window * self.window_height), center)
            if points[0].shape[0] > 0:
                center = np.mean(points[1])
                all_points = np.hstack((all_points[0], points[0])), np.hstack((all_points[1], points[1]))
                
        return all_points



def compute_curvature(fit, img, ym_per_pix = 1):
    yval = img.shape[0]
    curverad = ((1 + (2* fit[0]* yval * ym_per_pix + fit[1])**2)**1.5) / np.absolute(2 * fit[0])
    return curverad


def compute_x(fit, yval):
    return fit[0]*yval**2 + fit[1]*yval + fit[2]


def vechile_position(left_fit, right_fit, image_shape, xm_per_pix):
    yval = image_shape[0]
    image_center = image_shape[1] / 2
    x_left, x_right = compute_x(left_fit, yval), compute_x(right_fit, yval)
    lane_center = (x_left + x_right) / 2
    return (image_center - lane_center) * xm_per_pix


def draw_back(warped, undist, left_fit, right_fit, Minv):
    # Create an image to draw the lines on
    # warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.zeros((warped.shape[0], warped.shape[1], 3)).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped.shape[0] - 1, num = warped.shape[0])
    left_fitx = compute_x(left_fit, ploty)
    right_fitx = compute_x(right_fit, ploty)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    pts = np.int_([pts])
    # print("shape is")
    # print(pts.shape)
    cv2.fillPoly(color_warp, pts, (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    return result


def get_histogram(img):
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    return histogram

def find_peaks(histogram):
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base


def search_around(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
            # to self.recent_fit
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                              (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                               (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

            # Again, extract left and right line pixel positions
    allx = nonzerox[left_lane_inds], nonzerox[right_lane_inds]
    ally = nonzeroy[left_lane_inds], nonzeroy[right_lane_inds]
    return allx, ally
  

class Line(object):
    def __init__(self):
        # number of iteration to keep
        self.num_iter_to_keep = 1
        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = ([], [])  # should be the one at the bottom

        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = (np.array([False]), np.array([False]))
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # polynomial coefficients for the last n fit
        self.recent_fit = ([np.array([False])], [np.array([False])])
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = (np.array([0, 0, 0], dtype='float'), np.array([0, 0, 0], dtype='float'))
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        self.undist = None
        self.warped = None
        self.lost_frame_counter = 100000
        self.track_need_reinit = False
        self.img = None
        self.edge_image = None
        self.frame_counter = 0

    def feed_img(self, img):
        logging.debug("image received, frame {}".format(self.frame_counter))
        self.frame_counter += 1
        self.img = img
        self.undist = cv2.undistort(img, mtx, dist, None, mtx)

        self.warped = transform(self.undist, M)
        
        edge_image = detect_edge_v2(self.warped)
        self.edge_image = edge_image
        binary_warped = edge_image  # combine edges detected by different channels together
        logging.debug("lane line detected in last frame: {}".format(self.detected))
        logging.debug("number of lost frame counter: {}".format(self.lost_frame_counter))
        if self.detected or self.lost_frame_counter < 15:
            logging.debug("search around previous finding")
            self.allx, self.ally = search_around(binary_warped, self.current_fit[0], self.current_fit[1])

        else:
            self.track_need_reinit = True
            histogram = get_histogram(edge_image)

            ## much easier way find the peaks
            peaks = find_peaks(histogram)
            logging.debug("peaks founded: {}, {}".format(peaks[0], peaks[1]))

            left_points_detector_obj = CPointsDetector(edge_image, peaks[0], 18)
            right_points_detector_obj = CPointsDetector(edge_image, peaks[1], 18)
            points_left = left_points_detector_obj.get_points()
            points_right = right_points_detector_obj.get_points()
            self.ally = points_left[0], points_right[0]
            self.allx = points_left[1], points_right[1]
            
            # Fit a second order polynomial to each
        if self.sanity_check():
            if self.track_need_reinit:
                self.re_init_track_info()

            logging.debug("passed sanity check, update track_info now")
            self.update_track_info()

            self.detected = True
            # reset lost_frame_counter
            self.lost_frame_counter = 0

        else:
            logging.debug("did not pass sanity check")
            self.detected = False
            self.lost_frame_counter += 1

    def get_frame_couner(self):
        return self.frame_counter

    def get_curvature(self):
        # print("curvature")
        # print(tuple(self.best_fit))
        if self.lost_frame_counter >= 10000:
            return (0, 0)
        return (compute_curvature(fit, self.undist, ym_per_pix=ym_per_pix) for fit in self.best_fit)

    def get_draw_back_img(self):
        if self.lost_frame_counter >= 10000:
            return self.img

        back_image = draw_back(self.warped, self.undist, self.best_fit[0], self.best_fit[1], Minv)
        return back_image

    def get_edge_img(self):
        return self.edge_image

    def get_warp_img(self):
        return self.warped

    def get_points(self):
        return self.ally, self.allx

    def get_vehicle_position(self):
        if self.lost_frame_counter >= 10000:
            return 1000
        left_lane_position, right_lane_position = (sum(xfitted) / len(xfitted) for xfitted in self.recent_xfitted)
        lane_center = (left_lane_position + right_lane_position) / 2
        car_center = self.undist.shape[1] / 2
        self.lane_base_pos = (car_center - lane_center) * xm_per_pix
        return self.lane_base_pos

    def sanity_check(self):
        if self.ally[0].shape[0] == 0 or self.ally[1].shape[0] == 0:
            logging.warning("no points detected")
            return False
        m_left_fit = np.polyfit(self.ally[0] * ym_per_pix, self.allx[0] * xm_per_pix, 2)
        logging.debug(
            "fit for left lane line in meters: {}, {}, {} ".format(m_left_fit[0], m_left_fit[1], m_left_fit[2]))
        m_right_fit = np.polyfit(self.ally[1] * ym_per_pix, self.allx[1] * xm_per_pix, 2)
        logging.debug(
            "fit for righht lane line in meters: {}, {}, {}".format(m_right_fit[0], m_right_fit[1], m_right_fit[2]))

        self.current_fit = np.polyfit(self.ally[0], self.allx[0], 2), np.polyfit(self.ally[1], self.allx[1], 2)
        logging.debug("current fit in pixel space is: {}, {}".format(self.current_fit[0], self.current_fit[1]))
        x_left, x_right = compute_x(self.current_fit[0], self.img.shape[0]), compute_x(self.current_fit[1],
                                                                                       img.shape[0])
        logging.debug("x position is: {}".format((x_left, x_right)))
        if self.lost_frame_counter < 5:

            if np.absolute(x_left - self.recent_xfitted[0][-1]) > 100 or np.absolute(x_right - self.recent_xfitted[1][-1]) > 100:
                logging.debug("x position differ too much, rejected")
                return False

        m_left_curvature = compute_curvature(m_left_fit, img, ym_per_pix=ym_per_pix)
        m_right_curvature = compute_curvature(m_right_fit, img, ym_per_pix=ym_per_pix)

        self.radius_of_curvature = m_left_curvature, m_right_curvature
        logging.debug("curvature is: ({}, {})".format(m_left_curvature, m_right_curvature))
        # the curvature are in the same order of maginitude
        ratio = m_left_curvature / m_right_curvature
        if ratio > 20 or ratio < 0.05:
            return False

        # the distance between lines are right 
        # the lines are roughly parallel
        ploty = np.arange(0, self.undist.shape[0]) * ym_per_pix
        leftx = m_left_fit[0] * ploty ** 2 + m_left_fit[1] * ploty + m_left_fit[2]
        rightx = m_right_fit[0] * ploty ** 2 + m_right_fit[1] * ploty + m_right_fit[2]
        width = rightx - leftx
        logging.debug("min lane line width is: {}".format(np.min(width)))
        logging.debug("max lane line width is: {}".format(np.max(width)))
        if np.any(width > 4.4) or np.any(width < 2):
            return False
        width_length = width.shape[0]
        diff = width[10 : width_length - 1] - width[0 : width_length - 11]
        if np.any(diff > 0) and np.any(diff < 0):
            return False

        return True

    def update_track_info(self):
        if len(self.recent_fit[0]) > 0:
            self.diffs = self.current_fit[0] - self.recent_fit[0][-1], self.current_fit[1] - self.recent_fit[1][-1]
            if self.differ_too_much():
                logging.debug("fit differ with previous too much")
                self.re_init_track_info()

        self.recent_fit[0].append(self.current_fit[0])
        self.recent_fit[1].append(self.current_fit[1])
    
        x_left, x_right = compute_x(self.current_fit[0], self.img.shape[0]), compute_x(self.current_fit[1], img.shape[0])

        self.recent_xfitted[0].append(x_left)
        self.recent_xfitted[1].append(x_right)

        # remove data that are too old
        if len(self.recent_xfitted[0]) > self.num_iter_to_keep:
            self.recent_xfitted[0].remove(self.recent_xfitted[0][0])
            self.recent_xfitted[1].remove(self.recent_xfitted[1][0])
            self.recent_fit[0].remove(self.recent_fit[0][0])
            self.recent_fit[1].remove(self.recent_fit[1][0])
            # update bestx
        self.bestx = tuple(sum(x) / len(x) for x in self.recent_xfitted)
 
    
        # update bestfit
    
        self.best_fit = np.array(self.recent_fit[0]).mean(axis=0), np.array(self.recent_fit[1]).mean(axis=0)


    def re_init_track_info(self):
        logging.debug("reinit track info")
        self.recent_xfitted = ([], [])
        self.recent_fit = ([], [])
        self.bestx = None
        self.best_fit = (np.array([False]), np.array([False]))
        self.lost_frame_counter = 100000
        self.track_need_reinit = False


    def differ_too_much(self):
        thresh = np.array([0.00003, 0.05, 10])
        logging.debug("differ with previous fit is: {}, {}".format(self.diffs[0], self.diffs[1]))
        if np.any(np.absolute(self.diffs) > thresh):
            return True
        return False



# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[36]:

# In[152]:






# In[40]:




# In[112]:

line = Line()
def process_image2(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    line.feed_img(image)
    result = line.get_draw_back_img()
    left_curvature, right_curvature = line.get_curvature()
    position = line.get_vehicle_position()
    frame_counter = line.get_frame_couner()
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(result, "Frame {}".format(frame_counter), (10, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of curvature (Left)  = %.2f m' % (left_curvature), (10, 70), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of curvature (Right) = %.2f m' % (right_curvature), (10, 100), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    if position > 0:
        cv2.putText(result, 'Vehicle position : = %.2f m to the right of center  ' % (np.absolute(position)),
                (10, 130), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(result, 'Vehicle position : = %.2f m to the left of center  ' % (np.absolute(position)),
                    (10, 130), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result

def process_image3(image):
    line.feed_img(image)
    result = line.get_edge_img() * 255
    return np.dstack((np.zeros_like(result), result, np.zeros_like(result)))

import arrow


def process_image4(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    line.feed_img(image)
    result = line.get_draw_back_img()
    left_curvature, right_curvature = line.get_curvature()
    position = line.get_vehicle_position()
    frame_counter = line.get_frame_couner()
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(result, "Frame {}".format(frame_counter), (10, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of curvature (Left)  = %.2f m' % (left_curvature), (10, 70), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Radius of curvature (Right) = %.2f m' % (right_curvature), (10, 100), font, 1,
                (255, 255, 255), 2, cv2.LINE_AA)
    if position > 0:
        cv2.putText(result, 'Vehicle position : = %.2f m to the right of center  ' % (np.absolute(position)),
                (10, 130), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(result, 'Vehicle position : = %.2f m to the left of center  ' % (np.absolute(position)),
                    (10, 130), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    warped = line.get_warp_img()
    edge = line.get_edge_img()
    edge_zeros = np.zeros_like(edge)
    edge_img = np.dstack((edge_zeros, edge, edge_zeros)) * 255
    ally, allx = line.get_points()
    points_img = draw_points(warped, ally, allx, (255, 0, 0))

    row1 = np.hstack((warped, result))
    row2 = np.hstack((points_img, edge_img))
    final = np.vstack((row1, row2))

    return final





# In[38]:


    


# In[39]:




# In[57]:

# In[247]:


def draw_region(img):
    color_warp = np.zeros_like(img)
    pts = np.array([[(0, img.shape[0]),
                                             ((img.shape[1] * 0.50), (img.shape[0] * 0.60)),
                                         ((img.shape[1] * 0.53), (img.shape[0] * 0.60)),

                                          (img.shape[1], img.shape[0])]], dtype = np.int32)

        # Draw the lane onto the warped blank image
    pts = np.int_([pts])
        # print("shape is")
        # print(pts.shape)
    cv2.fillPoly(color_warp, pts, (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        # Combine the result with the original image
    result = cv2.addWeighted(img, 1, color_warp, 0.3, 0)
        # plt.imshow(result)
    return result


def num_generator_func():
    n = 0
    while True:

        yield n
        n += 1

num_generator = num_generator_func()

def process_image(image):
    scipy.misc.imsave("video_images/imgv{}.jpg".format(next(num_generator)), image, format=None)
    return image


def video2img(name):

    file_in = name + ".mp4"

    clip1 = VideoFileClip(file_in)
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile("rubbish.mp4", audio=False)

def edge_video():
    name = "challenge_video"
    file_in = name + ".mp4"
    file_out = name + "_out.mp4"
    clip1 = VideoFileClip(file_in)
    white_clip = clip1.fl_image(process_image3)  # NOTE: this function expects color images!!
    white_clip.write_videofile(file_out, audio=False)

def main():
    name = "project_video"
    file_in = name + ".mp4"
    file_out = name + "_out.mp4"
    clip1 = VideoFileClip(file_in)
    white_clip = clip1.fl_image(process_image2)  # NOTE: this function expects color images!!
    white_clip.write_videofile(file_out, audio=False)

def analyse():
    name = "harder_challenge_video"
    file_in = name + ".mp4"
    file_out = name + "_out.mp4"
    clip1 = VideoFileClip(file_in)
    white_clip = clip1.fl_image(process_image4)  # NOTE: this function expects color images!!
    white_clip.write_videofile(file_out, audio=False)

def main2():
    images = glob.glob('./video_images/*.jpg')
    # Step through the list and search for chessboard corners
    line = Line()
    for fname in images:
        img = imread(fname)
        line.feed_img(img)
        now = arrow.utcnow().timestamp
        edge = line.get_edge_img() * 255
        edge = np.dstack((edge, edge, edge))
        back = line.get_draw_back_img()
        scipy.misc.imsave("./edge/{}.jpg".format(now), edge)
        scipy.misc.imsave("./out/{}.jpg".format(now), back)

def test_region():
    images = glob.glob('./harder_video_images/img*.jpg')
    # Step through the list and search for chessboard corners

    for fname in images:
        img = imread(fname)
        region = draw_region(img)
        name = fname.replace("harder_video_images", "harder_region")
        scipy.misc.imsave("{}".format(name), region)
        # scipy.misc.imsave("./out/{}.jpg".format(now), back)

if __name__ == "__main__":
    # video2img("project_video")
    # test_region()
    # edge_video()
    # analyse()
    pass





