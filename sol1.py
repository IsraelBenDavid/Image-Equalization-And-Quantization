import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from imageio import imread
import scipy.cluster.vq as vq

# #### CONSTANTS #### #
GRAYSCALE = 1
RGB = 2
COLOR_MATRIX_DIMS = 3
GREATEST_VALUE = 255
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as a np.float64 matrix normalized to [0,1]
    """
    image = imread(filename)
    image = image.astype(np.float64)
    image /= GREATEST_VALUE
    if representation == GRAYSCALE:
        image = skimage.color.rgb2gray(image)
    return image


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    image = read_image(filename, representation)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    image = np.tensordot(imRGB, RGB_YIQ_TRANSFORMATION_MATRIX, axes=(2, 1))
    return image


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    image = np.tensordot(imYIQ, np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX), axes=(2, 1))
    return image


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return [im_eq, hist_orig, hist_eq]
    """
    if im_orig.ndim == COLOR_MATRIX_DIMS:
        # if the image is RGB run the alg` on channel y:
        hist_eq_on_y = _do_histogram_equalization(_get_y_channel(im_orig))
        # replace channel y and return
        return [_set_y_channel(im_orig, hist_eq_on_y[0]), hist_eq_on_y[1], hist_eq_on_y[2]]
    return _do_histogram_equalization(im_orig)


# Implementing the histogram equalization algorithm for one channel image (grayscale or channel y)
def _do_histogram_equalization(im_orig):
    hist_orig, bounds = np.histogram(im_orig * GREATEST_VALUE, bins=GREATEST_VALUE + 1)
    hist = np.cumsum(hist_orig)
    first_nonzero = np.nonzero(hist)[0][0]
    # equalization formula:
    hist = ((hist - hist[first_nonzero]) / (hist[GREATEST_VALUE] - hist[first_nonzero])) * GREATEST_VALUE
    hist = np.round(hist)
    # mapping the values:
    im_eq = hist[np.round(im_orig * GREATEST_VALUE).astype(int)]
    # creating new histogram
    # to_hist = im_eq * GREATEST_VALUE
    hist_eq, bounds = np.histogram(im_eq, bins=GREATEST_VALUE + 1)
    return [im_eq / GREATEST_VALUE, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iim_origter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    if im_orig.ndim == COLOR_MATRIX_DIMS:
        # if the image is RGB run the alg` on channel y:
        quantization_on_y = _do_quantization(_get_y_channel(im_orig), n_quant, n_iter)
        # replace channel y and return
        return _set_y_channel(im_orig, quantization_on_y[0]), quantization_on_y[1]
    return _do_quantization(im_orig, n_quant, n_iter)


# Implementing the quantization algorithm for one channel image (grayscale or channel y)
def _do_quantization(im_orig, n_quant, n_iter):
    segments = _calculate_segments(im_orig, n_quant)
    error = []
    quants = []
    for i in range(n_iter):
        quants = _calculate_quants(im_orig, segments)
        updated_segments = _update_segments(quants)
        error.append(_calculate_error(im_orig, quants, segments))
        if updated_segments == segments:
            break
        segments = updated_segments

    quanted_img = _map_to_quants(im_orig, segments, quants)
    return [quanted_img / GREATEST_VALUE, error]

# returns the Y channel of the given RGB image
def _get_y_channel(img):
    return rgb2yiq(img)[:, :, 0]


# sets the Y channel of the given RGB image
def _set_y_channel(img, y_to_set):
    temp = rgb2yiq(img)
    temp[:, :, 0] = y_to_set
    return yiq2rgb(temp)


# calculates the quantization error according to the formula
def _calculate_error(img, quants, segments):
    splitted_hist = _make_splitted_histogram(img, segments)
    segments_values = _make_segments(segments)
    error = 0
    for i in range(len(quants)):
        a = quants[i] - segments_values[i + 1]
        a = np.multiply(a, a)
        q = np.inner(splitted_hist[i + 1], a)
        error += q
    return error


# maps the image to the given quants according to the matching segments
def _map_to_quants(img, segments, quants):
    segments_values = _make_segments(segments)
    q_map = [np.array([quants[i]] * segments_values[i+1].shape[0]) for i in range(len(quants))]
    q_map = np.concatenate(q_map)
    ret = q_map[np.round(img * GREATEST_VALUE).astype(int)]
    return ret


# calculates the quants according to the segments bound and the image histogram
def _calculate_quants(img, segments_bounds):
    segments_values = _make_segments(segments_bounds)
    splitted_hist = _make_splitted_histogram(img, segments_bounds)
    quants = []
    for i in range(len(segments_bounds) - 1):
        if splitted_hist[i + 1].size == 0:
            quants.append(quants[-1])
            continue
        q = np.inner(splitted_hist[i + 1], segments_values[i + 1])
        q = q / np.sum(splitted_hist[i + 1])
        quants.append(q)

    return quants


# updates the segments according to the formula
def _update_segments(quants):
    segments = [0]
    for i in range(1, len(quants)):
        segments.append(int((quants[i - 1] + quants[i]) / 2))
    segments.append(GREATEST_VALUE)
    return segments


# calculates the segments bounds for the first iteration
def _calculate_segments(img, segments_num):
    hist_orig, bounds = np.histogram(img * GREATEST_VALUE, bins=GREATEST_VALUE + 1)
    hist_sum = np.cumsum(hist_orig)
    segment_size = int(hist_sum[-1] / segments_num)
    segments_list = []
    min_pixels = 0
    for i in range(segments_num + 1):
        required_index = np.where(hist_sum >= min_pixels)[0][0]
        segments_list.append(required_index)
        min_pixels += segment_size
    segments_list[-1] = GREATEST_VALUE  # sets the last bound to be 255
    return segments_list


# receives a list of segments bounds [0...255] (z) and returns matrix of segments values
def _make_segments(segments_bounds):
    segments = [val + 1 for val in segments_bounds]
    segments[0] = 0
    segments_values = np.arange(GREATEST_VALUE + 1)
    segments_values = np.split(segments_values, segments)
    return segments_values


# splits the image`s histogram to segments
def _make_splitted_histogram(img, segments_bounds):
    segments = [val + 1 for val in segments_bounds]
    segments[0] = 0
    hist_orig, bounds = np.histogram(img * GREATEST_VALUE, bins=GREATEST_VALUE + 1)
    hist = np.split(hist_orig, segments)
    return hist


def quantize_rgb(im_orig, n_quant):  # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    rows = im_orig.shape[0] * im_orig.shape[1]  # for the reshape func
    reshaped_img = np.reshape(im_orig, (rows, 3))
    means = vq.kmeans(reshaped_img, n_quant)[0]
    v_map = vq.vq(reshaped_img, means)[0]
    mapped_vectors = means[v_map]
    final_image = np.reshape(mapped_vectors, im_orig.shape)
    return final_image