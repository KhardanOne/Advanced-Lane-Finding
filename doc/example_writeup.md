## Writeup

---

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

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
Its this document.

### Architecture
I created a few classes and files to manage the complexity. The ones worth mentioning are:
1. `main.py` - The entry point of the program. Initialize global constants, the camera, creates calibration test images, and processes road images and videos.
1. `camera.py` - Camera class.
1. `image_processor.py` - The main pipeline is in `cls.do()` function for both videos and stills. `cls.init()` stores the two `LineHistory` objects for videos.
1. `line.py` - Contains data for a single line for a single frame and some sanity checks.
1. `line_history.py` - Stores the most recent versions of a single line. Calculates rolling averages and performs sanity checks.
1. `preprocess_helpers.py` - Performs images manipulations like Canny, crop, warp, etc.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
For this I created a Camera class. See `src/camera.py` and `main()` function in `main.py`. 

The calibration happens in Camera.calibrate() function as described:
> I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is > fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of > coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

> I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Apart from the required steps I also added a `Camera.save()` and `Camera.load()` function to load/save calibration and perspective correction matrices to camera_calib.json file. If it finds this file, it skips the calibration. I did this to reduce the time requirement of development iterations. Delete the file if you want to force re-running the calibration. For details see `Camera.__init__()`.

### Pipeline (both single images and videos)
The main function of the pipeline is the `cls.do()` function in `image_processor.py`. It call other functions to perform functions in the following chapters as described below.

#### 1. Provide an example of a distortion-corrected image.

In Camera.undistort() the previously stored camera matrix and distortion coeffitiens are fed to `cv2.undistor()` function. For testing purposes I undistorted all the camera calibration images. See an example below and all other images in `output/image_test_undistortion` folder. To generate them again uncomment in the `generate_test_images` function call in `main.py`'s `main()` function. 

![Undistorted calibration image][undistorted.jpg]
Notice the black pixels in the bottom-left corner.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
See `combined_threshold_4()` function in `preprocess_helpers.py` file. I decided to use value of 255 instead of 1, because this way I was able to use OpenCV Image Viewer plugin in PyCharm to view data as an image.

I combined three methods here, then combined them. See below the color image.
1. First I converted the image to grayscale. Applied a Sobel(1, 0) to detect vertical edges, then took the absolute values to include both the dark-to-light and light-to-dark transitions, finally applied a threshold detection. See the green color on the image below.
1. Second method: I converted the original image to HSL and took only the S channel. I appliad the Sobel algorithm to detect vertical edges, than applied threshold detection. See the red colors in the below image.

As a final step I combined the two datasets to one with logical or, resulting in a grayscale image containing only 0s and 255s.

![Combined thresholds][color_binary.png]
The image is from the OpenCV Image Viewer plugin, which displays it in BGR order. This explains the swapped colors. Note: you can also notice the effect of camera distortion removal. See the black bars in the bottom-left and top-left corners.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
This step requires two main steps: first determining the transformation matrix and the the matrix for revers transformation, then applying those matrices to actual images when needed.

Step 1
First I selected four source points on the image as shown below. Their coordinates are:

    CFG['perspective_calib_pts'] = np.float32([[217, 719], [593, 450], [689, 450], [1112, 719]])

The destination points are calculated from the bottom points. They have the same x coordinates as the originals, but with y = 0 values (on the top of the image).

![Generating Perspective Transformation Matrices][generating_perspective_transform_matrices.png]
Note: the black area is a padding to allow for extra space for roads with big curvatures. It proved to be overkill. Half of this amount would be enough most of the time.

Note: if you want to regenerate the transform matrices, delete `camera_calib.json` and rerun `main.py`.

I used `cv2.getPerspectiveTransform()` function inside `get_perspective_transform()` go generate both the forward and inverse matrices, which are later saved to file (and also kept in memory).

Step 2
I applied the perspective transform in the `warp()` function in `preprocess_helpers.py`. 
Before Warp:
![Before Warp][before_warp.png]

After Warp:
![After Warp][after_warp.png]
Note: these are not actual images. I generated them only for these documentation. Below you will see actual examples.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I use two methods for finding lane pixels.
1. I select the relevant points from the binary image that are close to a previosly found lane line polynom if the previous sanity checks were passed (only for videos). See `find_lane_pixels_around_poly()` in `preprocess.py`.
1. Otherwise I first detect the two starting points by summararizing the active pixels in bottom halves of both left and right sides of the image, then taking their maximum values. After that I apply the sliding windows algorithm. 

![Histogram and Sliding Windows][sliding_windows.png]

The function returns the found points to `ImageProcessor.do()` which later passes them to `fit_polynomial()` functin (in preprocess_helpers.py). It finds second degree polynomials for both lines and returns their coefficients.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.


I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
