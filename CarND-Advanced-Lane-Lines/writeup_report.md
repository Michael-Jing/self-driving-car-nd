
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 14 through 47 of the file called `advanced_lane_finding.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![distortion corrected][./writeup_images/camera_calibration_img.jpg]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![distortion corrected][./writeup_images/camera_calibration_img.jpg]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 92 through 149 in `advanced_lane_finding.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![binary_image][./writeup_images/binary_image.jpg]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_transform_matrix`, which appears in lines 49 through 65 in the file `advanced_lane_finding.py`  The `(get_transform_matrix)` function takes as inputs an image (`img`), and compute the perspective transformation matrix for images of that size,  I chose to hardcode the source and destination points in the following manner:

```
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

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I use another function `transform` which take in an image and a transformation matrix to do perspective transformation.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![transformed][./writeup_images/transform.jpg]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

1. find peaks of left and right lane line
I used a function `get_histogram` to sum each column(only the bottom half) in the thresholded binary image, then another function `find_peaks` to find the column number ie x position with the most pixels turned on.

2. use sliding window to find points on the thresholded binary image to find points on the left and right lane line
I created a class `CPointsDetector` to find pixels on the lane lines. The idea is the same as demonstrated in the course. ie, sliding through bands of the binary image from bottom up, in each band, search through a region around a center determined by peaks or points in the previous band, find the window with the most pixels turned on and append the position of those turned on pixels to the final result. The detected points are show in the below picture with read color.

3. use the found points to fit a polynomial curve, this is quite straightforward with the `np.polyfit` function

![lane_line_pixels][./writeup_images/pixel_locating_draw_back.jpg]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I computed curvature in the function `compute_curvatuer`, and vehicle position in the function `vechile_position`, vehicle position is just the difference between lane center and image center.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `advanced_lane_finding.py` in the function `draw_back()`.  the result is shown in the upper right corner of the following image.

![map lane line back][./writeup_images/pixel_locating_draw_back.jpg]
---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)
I also worked the challenge video out [link to challenge video result](./challenge_video_out.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

1. There are some black lines and shadow areas on the road in the challenge video which will pollute the result, so I combined the result of Sobel function and color thresholding (pick out white and yellow pixels) with bitwise or,

2. The above mentioned bitwise or works well in most cases, but sometimes it will filter out lane lines when the detected pixels are not too much, so I added blur to the result of color thresholding before bitwise or with edge detected by sobel.

3. In order to choose the thresholding for max and min lane line width, and max allowed differ with previous fit, I used logging to record the values and then made my decision based on the recorded values.
