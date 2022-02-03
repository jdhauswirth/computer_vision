# image_transforms.py

# import required libraries
import cv2
import numpy as np

## License
#MIT License

# Copyright (c) 2022 JDHaus

#### **Author**: [Joel Hauswirth](https://eportfolio.mygreatlearning.com/joel-d-hauswirth)
####                              https://olympus1.mygreatlearning.com/r/1933826/679
##### **Date**:         2022/01/31
##### **Last Updated**: 2022/02/02

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def put_text_on_image(img_in, text='orig'):
    '''
    Function to place text on an image indicating the type of transformation applied to image
    :param img_in: image for text placement
    :param text: indicates type of transformation applied to the image
    :return img_out: resulting image labeled with the transformation type
    '''
    img_out = cv2.putText(img_in, text, (20, 75), cv2.FONT_HERSHEY_COMPLEX, 1.8, (255, 255, 255), 3)
    return img_out

def draw_bbox_rect(img_in, start=(100, 100), end=(200, 200), color=(0, 0, 255)):
    '''
        Function to perform place a bounding bax on an image.
        :param img_in: the source image
        :param coords: set of coordinates for pixel locations the source image  [x1, y1, x2, y2]
                        x, y pixel coordinates to start and end the bounding box
        :param end: the source image x, y pixel coordinates to start bbox
        :param color:  a 3-tuple of BGR values.  default is (0, 0, 255) RED
        :return: img_out, area: image out and the area of the bounding box
    '''

    # make a copy of the image
    img_out = img_in.copy()

    # start points, end points red
    img_out = cv2.rectangle(img_out, start, end, color, 2)  # red  aka rgb(255,0,0)
    area_bbox = (end[0] - start[0]) * (end[1] - start[1])
    img_out = cv2.putText(img_out, str(area_bbox)+'_pixels', (260, 720),
                          cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 3)
    print("Area_bbox:        ", area_bbox)

    return img_out, area_bbox


def transform_image(img_in, mode=None, ksize=None, t1=None, t2=None, sob_k=None):
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

    # future enhancement should use python .switch somehow to make more like select/case
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

        elif mode == 'hls':
            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2HLS)
            img_out = put_text_on_image(img_out, text=mode)
            print("Function converted image to hls - (hue lightness saturation).")

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
image = cv2.imread('resources/t_lazy_7_ranch.png')
print("original image read with shape:", image.shape, " for (height, width, channels)")
cv2.imshow("Original Image", image)

# make a copy and mark as original
imageC = image.copy()
imageOrigT  = put_text_on_image(imageC, text='orig')

# perform 7 different operations/transforms on the original image + text label
imageCopy1  = transform_image(image, mode='copy')
# add a rectangle bounding box to calc pixel area occupied by wagon
imageCopy1, area_wagon = draw_bbox_rect(imageCopy1, (180, 480), (690, 670),  (0, 255, 0))

imageGray2  = transform_image(image, mode='gray')
imageBlur3  = transform_image(image, mode='blur', ksize=(5, 5))
imageHSV4   = transform_image(image, mode='hsv')
imageHLS7  = transform_image(image,  mode='hls')
imageCanny5 = transform_image(image, mode='canny', t1=100, t2=100)
imageSobel6 = transform_image(image, mode='sobel', sob_k=3)



# define scale and source images for stack
scale = 0.8
img_list =  ([imageOrigT, imageCopy1,  imageGray2, imageBlur3],
            [imageHSV4, imageHLS7,  imageCanny5, imageSobel6])

# create the stack
imgStack = stack_images(scale, img_list)

# show stacked images in 2 x 4 panel as defined by the list above
cv2.imshow("Image_Composite", imgStack)

# write transformed photo to file
cv2.imwrite('/Users/jdhauswirth/PycharmProjects/OpenCV_transform_photo/outputs/t_photo_panel.png', imgStack)
print("Script wrote stacked image panel file to 'outputs' directory")

cv2.waitKey(0)
