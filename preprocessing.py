import numpy as np
import cv2
import matplotlib.pyplot as plt
import cvutils
import os
import sys
import tkinter as tk
from tkinter import filedialog


# Extracting features from training images
def trainproc():
    train_imgs1 = cvutils.imlist("train_images\\500")
    train_imgs2 = cvutils.imlist("train_images\\2000")
    k = 0
    for tr in train_imgs1:
        pth = tr

        # Reading the image
        out = "train\\500"
        img = cv2.imread(pth)

        # resizing
        img = cv2.resize(img, (1200, 512), interpolation=cv2.INTER_LINEAR)

        # Denoising image
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Converting to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # compute the median of the single channel pixel intensities
        v = np.median(img)
        sigma = 0.33
        # apply automatic Canny edge detection using the computed median
        lower = int(max([ 0, (1.0 - sigma) * v ]))
        upper = int(min([ 255, (1.0 + sigma) * v ]))
        img = cv2.Canny(img, lower, upper)

        # Extracting features
        id1 = img[ 195:195 + 170, 190:190 + 85 ]
        id2 = img[ 330:330 + 105, 720:720 + 105 ]
        id3 = img[ 320:320 + 90, 865:865 + 205 ]
        id4 = img[ 250:250 + 40, 1120:1120 + 40 ]
        id5 = img[ 5:5 + 405, 660:660 + 40 ]
        id6 = img[ 284:284 + 132, 1090:1090 + 90 ]

        # Saving the features
        ids = [ id1, id2, id3, id4, id5, id6 ]
        out1 = "\\demo" + str(k) + ".jpg"
        d = 1
        for i in ids:
            cv2.imwrite(out + "\\id%d" % d + out1, i)
            d = d + 1
        k = k + 1

    k = 0
    for tr in train_imgs2:
        pth = tr

        # Reading the image
        out = "train\\2000"
        img = cv2.imread(pth)

        # resizing
        img = cv2.resize(img, (1200, 512), interpolation=cv2.INTER_LINEAR)

        # Denoising image
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Converting to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # compute the median of the single channel pixel intensities
        v = np.median(img)
        sigma = 0.33
        # apply automatic Canny edge detection using the computed median
        lower = int(max([ 0, (1.0 - sigma) * v ]))
        upper = int(min([ 255, (1.0 + sigma) * v ]))
        img = cv2.Canny(img, lower, upper)

        # Extracting features
        id1 = img[ 195:195 + 165, 225:225 + 55 ]
        id2 = img[ 330:330 + 95, 760:760 + 90 ]
        id3 = img[ 335:335 + 80, 890:890 + 205 ]
        id4 = img[ 255:255 + 25, 1105:1105 + 53 ]
        id5 = img[ 10:10 + 480, 726:726 + 35 ]
        id6 = img[ 280:280 + 140, 1100:1100 + 75 ]

        # Saving the features
        ids = [ id1, id2, id3, id4, id5, id6 ]
        out1 = "\\demo" + str(k) + ".jpg"
        d = 1
        for i in ids:
            cv2.imwrite(out + "\\id%d" % d + out1, i)
            d = d + 1
        k = k + 1


# Extracting features from test image# Extracting features from test image
def testproc():
    root = tk.Tk()
    pth = tk.filedialog.askopenfilename()
    root.destroy()
    out1 = "test\\ids1"
    out2 = "test\\ids2"

    # Reading the image
    img = cv2.imread(pth)
    cv2.imshow('qq', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # resizing
    img = cv2.resize(img, (1200, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('qq', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Denoising image
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Converting to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('qq', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # compute the median of the single channel pixel intensities
    v = np.median(img)
    sigma = 0.33
    # apply automatic Canny edge detection using the computed median
    lower = int(max([ 0, (1.0 - sigma) * v ]))
    upper = int(min([ 255, (1.0 + sigma) * v ]))
    img = cv2.Canny(img, lower, upper)
    cv2.imshow('qq', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extracting features
    id1 = img[ 195:195 + 170, 190:190 + 85 ]
    id2 = img[ 330:330 + 105, 720:720 + 105 ]
    id3 = img[ 320:320 + 90, 865:865 + 205 ]
    id4 = img[ 250:250 + 40, 1120:1120 + 40 ]
    id5 = img[ 5:5 + 405, 660:660 + 40 ]
    id6 = img[ 284:284 + 132, 1090:1090 + 90 ]
    ids1 = [ id1, id2, id3, id4, id5, id6 ]

    # Saving the features

    id1 = img[ 195:195 + 165, 225:225 + 55 ]
    id2 = img[ 330:330 + 95, 760:760 + 90 ]
    id3 = img[ 335:335 + 80, 890:890 + 205 ]
    id4 = img[ 255:255 + 25, 1105:1105 + 53 ]
    id5 = img[ 10:10 + 480, 726:726 + 35 ]
    id6 = img[ 280:280 + 140, 1100:1100 + 75 ]
    ids2 = [ id1, id2, id3, id4, id5, id6 ]

    d = 1
    for i in ids1:
        cv2.imwrite(out1 + "\\test%d.jpg" % d, i)
        d = d + 1
    d = 1
    for i in ids2:
        cv2.imwrite(out2 + "\\test%d.jpg" % d, i)
        d = d + 1

    # Displaying th features
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(ids1[ i ])
        plt.xticks([ ])
        plt.yticks([ ])

    plt.show()

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(ids2[ i ])
        plt.xticks([ ])
        plt.yticks([ ])

    plt.show()


# Main procedure
while True:
    x = input("Enter 0 to start extracting features from training images or 1 for testing the image\n")
    if x == '0':
        trainproc()
        print("Features extracted!!!!!!")
        os.system("perform-training.py")
        break
    if x == '1':
        testproc()
        print("Features extracted!!!!!!")
        os.system("perform-testing.py")
        break
    if x == 'exit':
        print("Exited!!!")
        break
    else:
        print("Enter correct key")
        continue
