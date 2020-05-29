#For image processing
import cv2
# To performing path manipulations 
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# For plotting
import matplotlib.pyplot as plt
# For array manipulations
import numpy as np
# For saving histogram values
from sklearn.externals import joblib
# Utility Package
import cvutils

# Store the path of training images in train_images
train_images500 = []
for d in range(6):
    i = d+1
    ti = cvutils.imlist("train/500/id%i"%i)
    train_images500.append(ti)
n = len(train_images500[0])

train_images2000 = []
for d in range(6):
    i = d+1
    ti = cvutils.imlist("train/2000/id%i"%i)
    train_images2000.append(ti)
n = len(train_images2000[0])

X_test500 = []
X_test2000 = []

# For each image in the training set calculate the LBP histogram
# and update X_test, X_name and y_test
for train_image in train_images500:
    # Read the image
    X_temp = []
    for i in range(n):
        im = cv2.imread(train_image[i])
        # Convert to grayscale as LBP works on grayscale image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        radius = 3
        # Number of points to be considered as neighbours
        no_points = 8 * radius
        # Uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        # Calculate the histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        X_temp.append(hist)
        # Append histogram to X_test
    X_test500.append(X_temp)

for train_image in train_images2000:
    # Read the image
    X_temp = []
    for i in range(n):
        im = cv2.imread(train_image[i])
        # Convert to grayscale as LBP works on grayscale image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #im_gray = cv2.resize(im_gray, (100, 100), interpolation=cv2.INTER_LINEAR)
        radius = 3
        # Number of points to be considered as neighbours
        no_points = 8 * radius
        # Uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        # Calculate the histogram
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        X_temp.append(hist)
        # Append histogram to X_test
    X_test2000.append(X_temp)
# Dump the  data
joblib.dump((X_test500, X_test2000,n), "lbp.pkl", compress=3)
 
print("Images are been trained")
os.system("preprocessing.py")
