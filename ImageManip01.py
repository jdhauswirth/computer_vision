import cv2
import numpy as np


def put_text_on_image(img_in, text=None):
    '''
    Function to place text on an image indicating the type of transformation applied to image

    :param img_in: image for text placement
    :param text: indicates type of transformation applied to the image

    :return img_out: resulting image labeled with the transformation type
    '''
    img_out = cv2.putText(img_in, text, (20, 75), cv2.FONT_HERSHEY_COMPLEX, 1.8, (255, 255, 255), 3)
    return img_out

def manipulate_image(img_in, mode=None, ksize=None, t1=None, t2=None, sob_k=None):
    '''
    Function to perform a variety of image transformations such as:

    Arguments:

    :param img_in: the source image
    :param mode: copy, gray, blur, hsv, canny, sobel8u
    :param ksize: kernel size for the Gaussian blur function
    :param t1: threshold 1 for the Canny function:
    :param t2: threshold 2 for the Canny function:
    :param sob_k: sobel kernel size

    :return: img_out:
    '''
    if mode != None:

        # make a copy of the image
        img_out = img_in.copy()

        if mode == 'copy':
            img_out = put_text_on_image(img_out, text=mode)
            print("Function made a copy of image.")

        elif mode == 'gray':
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
            img_out = put_text_on_image(img_out, text=mode)
            print("Function converted image to grayscale.")

        elif mode == 'blur':
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
            img_out = cv2.GaussianBlur(img_out, (ksize), 1)
            img_out = put_text_on_image(img_out, text=mode)
            print("Function converted image with Gaussian Blur.")

        elif mode == 'hsv':
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)
            img_out = put_text_on_image(img_out, text=mode)
            print("Function converted image to Hue, Saturation and Values.")

        elif mode == 'canny':
            img_out = cv2.Canny(img_out, t1, t2)
            img_out = put_text_on_image(img_out, text=mode)
            print("Function converted image to show Canny edges with thresholds at (", t1, ",", t2, ").")

        elif mode == 'sobel':
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
            img_out = cv2.Sobel(img_out, cv2.CV_8U, 1, 0, ksize=sob_k)
            img_out = put_text_on_image(img_out, text=mode)
            print("Function converted image to show Sobel edges with kernel size", sob_k, ".")
    else:
        img_out = np.NaN
        print("Function did nothing!")

    return img_out

def stack_images(scale, imgArray):
    '''
    Function to resize and arrange images into a single window.
    ** Credit and thanks to M. Hassan at Computer Vision Zone for this original code for this function.
    https://www.computervision.zone/courses/ **

    :param scale: decimal number defining factor to impact resulting window size.
    :type scale: float
    :param imgArray:   a python list defining array of images.
    :type imgArray: list
    :return: formatted images in a pane
    :type return: a composite window of the stacked images
    '''
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],
                                                (0, 0),
                                                None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y],
                                                (imgArray[0][0].shape[1],
                                                 imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# import original image
image = cv2.imread('Resources/t_lazy_7_ranch.png')
print("original image shape:", image.shape)
cv2.imshow("Original Image", image)

# perform 6 different operations on the original image
imageCopy1 = manipulate_image(image, mode='copy')
imageGray2 = manipulate_image(image, mode='gray')
imageBlur3 = manipulate_image(image, mode='blur', ksize=(11, 11))
imageHSV4 = manipulate_image(image, mode='hsv')
imageCanny5 = manipulate_image(image, mode='canny', t1=100, t2=100)
imageSobel6 = manipulate_image(image, mode='sobel', sob_k=3)

# define scale and source images for stack
scale = 0.8
img_list = [imageCopy1, imageGray2, imageBlur3], [imageHSV4, imageCanny5, imageSobel6]

# create the stack
imgStack = stack_images(scale, img_list)

# show stacked images in 3 x 2 panel as defined by the list above
cv2.imshow("Image_Composite", imgStack)

cv2.waitKey(0)
