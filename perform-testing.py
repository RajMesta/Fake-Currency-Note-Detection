# For image processing
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


# Displaying the fake result image
def fake_img():
    pth = "fake.jpg"
    img = cv2.imread(pth)
    cv2.imshow('FAKE!!!!', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Displaying the genuine result image
def genuine_img():
    pth = "genuine.jpg"
    img = cv2.imread(pth)
    cv2.imshow('GENUINE!!!!', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load the List for storing the LBP Histograms, address of images and the corresponding label
X_test500, X_test2000, n = joblib.load("lbp.pkl")

# Store the path of testing images in test_images
test_images500 = cvutils.imlist("test/ids1")
test_images2000 = cvutils.imlist("test/ids2")

# Dict containing scores
results_all500 = {}
results_all2000 = {}

# total scores
tot500 = 0
tot2000 = 0

for i in range(6):
    # Read the image
    im = cv2.imread(test_images500[ i ], 0)

    radius = 3
    # Number of points to be considered as neighbourers 
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im, no_points, radius, method='uniform')
    # Calculate the histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    # Display the query image
    results = [ ]
    scores = 0
    # For each image in the training dataset
    # Calculate the chi-squared distance and the sort the values
    for index, x in enumerate(X_test500[ i ]):
        score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.HISTCMP_CHISQR)
        #        print(score)
        scores += score
    scores = scores / 3
    results.append(round(scores, 3))
    results_all500[ "id%i" % i ] = results
    tot500 += results[ 0 ]
#   print(results_all)

for i in range(6):
    # Read the image
    im = cv2.imread(test_images2000[ i ], 0)

    radius = 3
    # Number of points to be considered as neighbourers
    no_points = 8 * radius
    # Uniform LBP is used
    lbp = local_binary_pattern(im, no_points, radius, method='uniform')
    # Calculate the histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    # hist = x[:, 1]/sum(x[:, 1])
    # Display the query image
    results = [ ]
    scores = 0
    # For each image in the training dataset
    # Calculate the chi-squared distance and the sort the values
    for index, x in enumerate(X_test2000[ i ]):
        score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.HISTCMP_CHISQR)
        #        print(score)
        scores += score
    scores = scores / 3
    results.append(round(scores, 3))
    results_all2000[ "id%i" % i ] = results
    tot2000 += results[ 0 ]
#   print(results_all)

buff1 = 0
buff2 = 0

while True:
    if results_all500[ 'id0' ][ 0 ] > 0.006:
        break
    if results_all500[ 'id1' ][ 0 ] > 0.02:
        break
    if results_all500[ 'id2' ][ 0 ] > 0.008:
        break
    if results_all500[ 'id3' ][ 0 ] > 0.06:
        break
    if results_all500[ 'id4' ][ 0 ] > 0.02:
        break
    if results_all500[ 'id5' ][ 0 ] > 0.07:
        break
    else:
        buff1 = 1
        break

while True:
    if results_all2000[ 'id0' ][ 0 ] > 0.006:
        break
    if results_all2000[ 'id1' ][ 0 ] > 0.02:
        break
    if results_all2000[ 'id2' ][ 0 ] > 0.008:
        break
    if results_all2000[ 'id3' ][ 0 ] > 0.06:
        break
    if results_all2000[ 'id4' ][ 0 ] > 0.02:
        break
    if results_all2000[ 'id5' ][ 0 ] > 0.07:
        break
    else:
        buff2 = 1
        break

if buff1 == 0 and buff2 == 0:
    fake_img()
    print("1")
elif buff1 and buff2:
    print("2")
    if tot2000 > tot500:
        print("500 CURRENCY NOTE")
    else:
        print("2000 CURRENCY NOTE")
    genuine_img()
elif buff1:
    print("3")
    print("500 CURRENCY NOTE")
    genuine_img()
else:
    print("4")
    print("2000 CURRENCY NOTE")
    genuine_img()

os.system("preprocessing.py")
