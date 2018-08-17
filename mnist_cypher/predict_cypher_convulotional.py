import keras
import sys
import cv2
import numpy as np
import scipy.ndimage as ndimage
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image
import math

def isMostlyBlack(img):
    pixels = img.getdata()  # get the pixels as a flattened sequence
    black_thresh = 50
    nblack = 0
    for pixel in pixels:
        if pixel < black_thresh:
            nblack += 1
    n = len(pixels)

    if (nblack / float(n)) > 0.5:
        return True


def removeEmptySpaceSingle(img):
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)

    return img

def removeEmptySpace(img, top_left, bottom_right):
    while np.sum(img[0]) == 0:
        top_left[0] += 1
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        top_left[1] += 1
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        bottom_right[0] -= 1
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        bottom_right[1] -= 1
        img = np.delete(img, -1, 1)
    return img, bottom_right, top_left


def fitOuterIn20x20(rows, cols, img) :
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))

    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')
    return img, rowsPadding, colsPadding

def getBestShiftForNumberInCenter(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shiftNumberToCenter(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def addBlackLinesTo28x28Image(img):
    rows, cols = img.shape
    compl_dif = abs(rows - cols)
    half_Sm = compl_dif / 2
    half_Big = half_Sm if half_Sm * 2 == compl_dif else half_Sm + 1
    if rows > cols:
        img = np.lib.pad(img, ((0, 0), (half_Sm, half_Big)), 'constant')
    else:
        img = np.lib.pad(img, ((half_Sm, half_Big), (0, 0)), 'constant')

    img = cv2.resize(img, (20, 20))
    img = np.lib.pad(img, ((4, 4), (4, 4)), 'constant')

    shiftx, shifty = getBestShiftForNumberInCenter(img)
    shifted = shiftNumberToCenter(img, shiftx, shifty)
    return shifted

def changeOneDimensionArrayToImageArrayDimensions(img):
    return np.arange(img).reshape(1, 784)
    #data3 = data3.reshape((data3.shape[0] * 3, 28, 28))

(train_x, train_y) , (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)
train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)
model = Sequential()
model.add(Dense(units=128,activation="relu",input_shape=(784,)))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=10,activation="softmax"))
model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.load_weights("mnistmodel.h5")
imagepath = sys.argv[1]

image_color = cv2.imread(imagepath)
rows, cols, color = image_color.shape
if(rows != cols):
    image_gray = cv2.imread(imagepath,0)
    #better black and white version
    (thresh, image_gray) = cv2.threshold(255 - image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #create matrix with shape of number, contains which place already found a digit so a digit doesnt get read twice
    digit_image = -np.ones(image_gray.shape)

    height, width = image_gray.shape

    # crop in to different images
    for cropped_width in range(100, 300, 20):
        for cropped_height in range(100, 300, 20):
            for shift_x in range(0, width-cropped_width, cropped_width/4):
                for shift_y in range(0, height-cropped_height, cropped_height/4):
                    image_gray_cropped = image_gray[shift_y:shift_y + cropped_height, shift_x:shift_x + cropped_width]

                    #if cropped image is almost empty, move to next
                    if np.count_nonzero(image_gray_cropped) <= 20:
                        continue

                    #if cut through digit, and thus no white border, continue
                    if (np.sum(image_gray_cropped[0]) != 0) or (np.sum(image_gray_cropped[:, 0]) != 0) or (np.sum(image_gray_cropped[-1]) != 0) or (np.sum(image_gray_cropped[:,
                                                                                                                                   -1]) != 0):
                        continue
                    #save top left and bottom right position of the rectangle
                    top_left = np.array([shift_y, shift_x])
                    bottom_right = np.array([shift_y + cropped_height, shift_x + cropped_width])

                    image_gray_cropped, bottom_right, top_left = removeEmptySpace(image_gray_cropped, top_left, bottom_right)

                    #Check if there is already a digit inside the current rectangle
                    actual_w_h = bottom_right - top_left
                    if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] + 1) >
                            0.2 * actual_w_h[0] * actual_w_h[1]):
                        continue

                    image_gray_cropped = addBlackLinesTo28x28Image(image_gray_cropped)

                    # reshape the image
                    image_gray_cropped = image_gray_cropped.reshape((1, 784))

                    img_class = model.predict_classes(image_gray_cropped)

                    digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = img_class[0]

                    cv2.rectangle(image_color, tuple(top_left[::-1]), tuple(bottom_right[::-1]), color=(0, 255, 0),
                                  thickness=2)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # digit we predicted
                    cv2.putText(image_color, str(img_class[0]), (top_left[1], bottom_right[0] + 50),
                                font, fontScale=0.8, color=(0, 255, 0), thickness=4)


    cv2.imwrite("digitized_image.png", image_color)
else:
    img = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(255 - img, (28, 28))
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # img = img.reshape((28,28))

    img_array = image.img_to_array(img)
    # if not isMostlyBlack(img) :
    #   img_array = cv2.resize(255 - img_array, (28, 28))

    # (thresh, img_array) = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_array = removeEmptySpaceSingle(img_array)

    (rows, cols) = img.shape
    img_array, rowsPadding, colsPadding = fitOuterIn20x20(rows=rows, cols=cols, img=img_array)
    # img_array = np.lib.pad(img_array,(rowsPadding,colsPadding),'constant')
    shiftx, shifty = getBestShiftForNumberInCenter(img_array)
    img_array = shiftNumberToCenter(img_array, shiftx, shifty)
    test_img = img_array.reshape((1, 784))

    img_class = model.predict_classes(test_img)
    # test_img = changeOneDimensionArrayToImageArrayDimensions(img_array)
    prediction = img_class[0]
    classname = img_class[0]
    print("Class: ", classname)



