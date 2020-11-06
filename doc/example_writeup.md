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

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
Its this document.

---

### Architecture
I created a few classes and files to manage the complexity. The ones worth mentioning are:
1. `main.py` - The entry point of the program. It initializes global constants and the camera, creates calibration test images and processes road images and videos.
1. `camera.py` - Camera class.
1. `image_processor.py` - The main pipeline is in `cls.do()` function for both videos and stills.
1. `line.py` - Contains data for a single line for a single frame. Plus some sanity checks.
1. `line_history.py` - Stores the 6 most recent versions of a single line. Calculates rolling averages and performs sanity checks.
1. `preprocess_helpers.py` - Performs image manipulations like crop, Canny, warp, etc.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
For this I created a Camera class. See `src/camera.py` and `main()` function in `main.py`. 

The calibration happens in Camera.calibrate() function as described:
> I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is > fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of > coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

> I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Apart from the required steps I also added a `Camera.save()` and `Camera.load()` function to load/save calibration and perspective correction matrices to camera_calib.json file. If it finds this file, it skips the calibration. I did this to reduce the time requirement of development iterations. Delete the file if you want to force re-running the calibration. For details see `Camera.__init__()`.

---
### Pipeline (both single images and videos)
In `main.py` the `process_still_images()` function iterates over all images and sends them to image processor (stills only).

The main function of the pipeline is the `ImageProcessor.do()` in `image_processor.py`. It calls other functions to perform functions described in the following chapters.

#### 1. Provide an example of a distortion-corrected image.
In `Camera.undistort()` the previously stored camera matrix and distortion coeffitiens are fed to `cv2.undistor()` function. For testing purposes I undistorted all the camera calibration images. See an example below. All other images can be found in `output/image_test_undistortion` folder. To generate them again just uncomment the `generate_test_images()` function call in `main()` function. 

![Undistorted calibration image](undistorted.jpg)
Notice the black pixels in the bottom-left corner.

---

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
See `combined_threshold_4()` function in `preprocess_helpers.py` file. I decided to use the value of 255 instead of 1, because this way I was able to use OpenCV Image Viewer plugin in PyCharm to view data as an image.

I combined two methods:
1. First I converted the image to grayscale. Applied a Sobel(1, 0) to detect vertical edges, then took the absolute values to include both the dark-to-light and light-to-dark transitions, and finally filtered by thresholds. See the green color on the image below.
1. Second method: I converted the original image to HSL and took only the S channel. I applied the Sobel algorithm to detect vertical edges, then applied filtered by thresholds. See the red colors in the below image.

As a final step I combined the two datasets to one with logical `or`, resulting in a grayscale image containing only 0s and 255s.

![Combined thresholds](color_binary.png)
The image is from the OpenCV Image Viewer plugin, which displays it in BGR order. This explains the swapped colors if you compare it to the code. <BR>
Note: you can also notice the effect of camera distortion removal. See the black bars in the bottom-left and top-left corners.


---

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
First I determined the transformation matrix and the the matrix for reverse transformation, then applied those matrices to actual images when needed.

**Step 3.1**
First I selected four source points on the image as shown below. Their coordinates are:

    CFG['perspective_calib_pts'] = np.float32([[217, 719], [593, 450], [689, 450], [1112, 719]])

The destination points are calculated from the bottom points. They have the same x coordinates as the bottom points, but with y = 0 values (on the top of the image).

![Generating Perspective Transformation Matrices](generating_perspective_transform_matrices.png)
Note: the black area is a padding to allow for extra space for roads with big curvatures.

Note: if you want to regenerate the transformation matrices, delete the `camera_calib.json` file and rerun `main.py`.

I used `cv2.getPerspectiveTransform()` function inside `get_perspective_transform()` to generate both the forward and inverse matrices, which were later saved to file.

**Step 3.2**
I applied the perspective transform in the `warp()` function in `preprocess_helpers.py`. 
Before Warp:
![Before Warp](before_warp.png)

After Warp:
![After Warp](after_warp.png)
Note: these are not actual images from the input folder. I generated them only for this documentation. Below you will see actual examples.

---

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I used two methods for finding lane pixels:
1. I selected relevant points from the binary image that were close to a previosly found lane line polynom, but only if the previous sanity checks were passed (only for videos). See `find_lane_pixels_around_poly()` in `preprocess.py`.
1. Otherwise I first detect the two starting points by summararizing the active pixels in bottom halves of both left and right sides of the image, then taking their maximum values. See the histogram on the below image. After that I apply the sliding windows algorithm. 

![Histogram and Sliding Windows](sliding_windows.png)

The function returns the found points to `ImageProcessor.do()` which later passes them to `fit_polynomial()` function (`in preprocess_helpers.py`). It finds second degree polynomials for both lines, and returns their coefficients.

---

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
Radius of curvature is calculated in `measure_radius_m()` function in `preprocess_helpers.py` with the following equation:

    left_radius_m  = ((1 + (2 * left_fit[0]  * y_eval_m + left_fit[1]) ** 2 ) ** 1.5) / np.absolute(2 * left_fit[0])

To get the car's offset from the lane center I calculated the values of polynoms at the bottom of the image, averaged the two values and finally subtracted the screen center's x coordinate. All units (including screen center coordinate) was previously transformed to meter units.

For unit conversions I assumed a lane width of 3.7 meters and estimated the length of the relevant are to be 30 meters. For real-word usage these values should and can easily be measured with the actual camera and car setup.

---

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
Once the left and right lane polynomials were found the `draw_polys_inplay()` (`in preprocess_helpers.py`) was called to display the green overlay. It does it by calling cv2.fillPoly() after converting the data to the required format.

![Poly on road](test2.jpg)

---

#### 7. Extra: verbose mode
The majority of the information gathered in the above steps can be drawn over the image. To do this set `verbose` to `True` in the first line of `ImageProcessor.do()`

Example image with verbose mode ON:
![Overlay](overlay.jpg)

---

#### 8. Extra: sanity checks
`ImageProcessor.sanity_checks()` function performs the following sanity checks:
* Parallel check: examines the distance between the lines at three different y coordinats and takes the difference between the max and min values. It is considered good below a certain treshold.
* Distance check: examines the distance between the lines at three different y coordinates. It is considered good if the maximum of the three distances is below a preset treshold.
* Prior check (videos only): compares the coefficiens of the current lines with the rolling average values of the previous frames. It is considered good if all values are below their preset values.

The results are used for the following purposes:
* a quality value is generated for the lines
* they can also be written to the image using verbose mode as seen above.

---

### Pipeline (extra steps with video)
In `main.py` the `process_video()` function does set up the very same `ImageProcessor.do()` as callback for VideoFileClip.fl_image(). 

The most notable extra functionality in `ImageProcessor.do()` is the history. The ImageProcessor contains two instances of LineHistory class. They are intantiated by the above `process_video()` function. Its purpose:
1. Provide information on prior line quality. This quality determines if `find_lane_pixels_sliding_window()` or `find_lane_pixels_around_poly()` functions will be used.
1. Provide polynomial coefficients for `find_lane_pixels_around_poly()`
1. Provide rolling averages of prior polynomial coefficients for smoothing purposes.


#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](../output/videos/project_video_verbose.mp4) with verbose ON. Or a clean one [without it](../output/videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The current version cannot handle the two challenge videos. I am planning to return to this topic after finishing the nanodegree, to improve the algorithm. The main issues:
1. The sliding windows algorithm cannot handle steep curves. Some other algorithm is needed. E.g. building a polygon buttom up from the same starting points but with the ability to change direction and go sideways. Needs experimentations.
1. The algorithm cannot cope with `challenge_video.mp4`'s strong longitudinal lines and faint lane lines. Needs more experimentations with other algorithms and tuning parameters.
1. The `harder_challenge_video.mp4`s steep curves most probably cannot be matched by a second degree polynomial. Needs experimenations with higher level polynomials and other algorithms.
1. Radius of curvature values fluctuate and deviate a lot. This is somewhat understandable, but can be improved by improving the overall quality and by also smoothing their values by using rolling averages.
1. Etc.
