## Vehicle Detection Pipeline

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

The entire code of the project can be found in the [VehicleDetection iPython notebook](./VehicleDetection.ipynb).

The HTML save of the notebook is [here](./VehicleDetection.html).

[//]: # (Image References)

[mosaic_cars]: ./mosaic_cars.png
[mosaic_noncars]: ./mosaic_noncars.png

[mosaic_cars_exp]: ./mosaic_cars_experiment.png
[mosaic_noncars_exp]: ./mosaic_noncars_experiment.png

[ex_final]: ./example_final.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images

The code for extracting HOG features (and optionally images) is confined to the helper function,
get_hog_features.


Here are some examples of cars and non-cars, in single channel, along with their HOG features extracted.

Cars:

![cars][mosaic_cars]

Non-cars:

![non-cars][mosaic_noncars]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Cars:

![cars][mosaic_cars_exp]

Non-cars:

![non-cars][mosaic_noncars_exp]

#### 2. Explain how you settled on your final choice of HOG parameters.

Parameter selection is by far the hardest and most frustrating part. It is really tempting to create
a meta-pipeline and then use some sort of optimization technique to search the parameter space, e.g.
from something as simple as using Twiddle algorithm to a more advanced, e.g. simulated annealing search.

But such an effort seems well beyond the scope of this project. It would also require a large body of well-labeled data to ascertain the quality of the selected parameters.

In lieu of all of this, I have been choosing parameters non-systematically, using the informal "this seems to yield results that suck a bit less" approach.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 80% / 20% split of training/testing data, and using StandardScaler to normalize the data. The C parameter of the linear SVM has been set to 0.01 from the default value of 1. This is the parameter that controls penalty for misclassified samples, and decreasing it can improve classifier's generalization / reduce overfitting.

### Sliding Window Search, Persistent Heatmap and Bounding Boxes

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have used multiple sliding window templates of various sizes, locations and overlaps, to loosely
mimic the perspective view of the road.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I ended up using HOG features of the Y-channel of the YCrCb colorspace.

For each frame, I used a sliding window technique and then used the boxes identified by the SVM classifier to update a heatmap.

The SVM decision is thresholded at 0.2 value of the decision function to reduce false positives.

The heatmap persisted from frame to frame, with previous value "fading" at a specified factor.

Then, I used `scipy.ndimage.measurements.label()` to identify blobs on that heatmap.

For for each blob on the heatmap above certain thresholded hotness and size, I found a bounding box for all the intersecting positives found by the SVM.

`slide_and_search()`, `make_heatmap()`, `get_heat_boxes()`, `expand_heat_boxes()` functions are
the building blocks of this approach, put together in the `detect_wehicles()` function.

This allowed to cut down on the false positives while approximating a bounding box for the entire vehicle.

In the image below, left to right, top to bottom:

* Image with the final bounding boxes
* All sliding windows
* All SVM positives
* Heatmap
* Single channel before HOG extraction
* HOG image

![various stages][ex_final]
---

### Video Implementation

#### 1. Final video is [here](./project_video_out.mp4)

Or see it [on YouTube](https://www.youtube.com/watch?v=U0ohqkaIWYU)

The frames are:

Top-left: original video with the final bounding box for located vehicle

Top-right: all sliding windows used in search

Low-left: raw positives from the SVM classifier

Low-right: A heatmap persisting from frame to frame, before thresholding by intensity and size

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipelines accuracy and performance leave much to be desired.

The accuracy can be improved by:

- extending the training set for the classifier. While the classifier's performance on the training set wasn't bad, the error rate of 4-5% means that we can expect a number of false positives and false negatives in any given frame. Improving the representativeness of the training data can help;

- tweaking feature extraction to achieve better accuracy; augmenting HOG features with other features;

- tweaking the heatmap heuristic for improving both the false positives and false negatives;

The performance can be improved by:

- extracting the HOG features for the entire image and then reusing them in the sliding window extraction.

- parallelizing sliding window search, taking advantage of multiple cores

- reducing the size of the feature vector extracted (hopefully without sacrificing the performance)

- reducing the number of windows in the sliding window approach. This will require tweaking the heatmaps technique parameters as well









