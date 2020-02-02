import cv2
import numpy as np


def get_gaussian_filter(sigma):
    size = int(6 * sigma + 1)
    kernel_x = cv2.getGaussianKernel(ksize=size, sigma=sigma)
    kernel_y = np.zeros((1, size))
    for i in range(size):
        kernel_y[0, i] = kernel_x[i, 0]
    kernel = kernel_y * kernel_x
    return kernel


def get_derivative_x(img):
    derivative_kernel = np.ones((1, 3))
    derivative_kernel[0, 2] = -1
    return img[:, :-2] - img[:, 2:]


def get_derivative_y(img):
    derivative_kernel = np.ones((3, 1))
    derivative_kernel[2, 0] = -1
    return img[:-2, :] - img[2:, :]


def threshold(img, thre):
    w = len(img)
    h = len(img[0])
    upper_mask = img >= thre
    ret = upper_mask * np.ones((w, h, 3)) * 255
    return ret


def main():
    # getting image
    img = cv2.imread("Books.jpg", cv2.IMREAD_GRAYSCALE)

    w = len(img)
    h = len(img[0])

    # make img a 3 channel image because the rest of the program is written with assumption that img is a 3 channeled image
    img1 = np.zeros((w, h, 3))
    img1[:, :, 0] = img
    img1[:, :, 1] = img
    img1[:, :, 2] = img
    img1 = img1.astype(np.uint8)
    img = img1

    # filter constraints
    gaussian_blurr_sigma = 3
    thre = 4
    
    gaussian_filter = get_gaussian_filter(gaussian_blurr_sigma)

    # making 2D filters derivative x and derivative y
    gaussian_derivative_filter_x = get_derivative_x(gaussian_filter)
    gaussian_derivative_filter_y = get_derivative_y(gaussian_filter)
    
    # calculating derivative_x and derivative_y
    derivative_y = cv2.filter2D(img, 1, gaussian_derivative_filter_y)
    derivative_x = cv2.filter2D(img, 1, gaussian_derivative_filter_x)

    derivative = np.sqrt((derivative_x * derivative_x) + (derivative_y * derivative_y))
    
    derivative = threshold(derivative, thre)

    # make derivative such that if a pixel is an edge pixel in a channel it is an edge pixel of the whole image
    b_channel = derivative[:, :, 0]
    g_channel = derivative[:, :, 1]
    r_channel = derivative[:, :, 2]
    white_b_mask = b_channel == 255
    white_g_mask = g_channel == 255
    white_r_mask = r_channel == 255
    h = len(derivative[0])
    w = len(derivative)
    derivative[:, :, 0] = white_b_mask * np.ones((w, h)) * 255 + white_g_mask * np.ones(
        (w, h)) * 255 + white_r_mask * np.ones((w, h)) * 255
    derivative[:, :, 1] = derivative[:, :, 0]
    derivative[:, :, 2] = derivative[:, :, 0]

    # save edge points and the gradient dircetion in every pixel
    derivative = derivative.astype(np.uint8)
    cv2.imwrite("EdgePoints.jpg", derivative)

    degree = np.rad2deg(np.arctan2(derivative_y, derivative_x))
    degree = (((degree < 0) * (degree + 180)) + ((degree >= 0) * degree))
    degree = degree.astype(np.uint8)
    cv2.imwrite("GradientDirection.jpg", degree)



if __name__ == '__main__':
    main()
