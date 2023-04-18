import matplotlib.pyplot as plt

from sol1 import read_image, histogram_equalize, quantize, quantize_rgb
from sol1 import GRAYSCALE, RGB


def grayscale_example(filename):
    # Reading a grayscale image
    im = read_image(filename, representation=GRAYSCALE)

    # Displaying the original image
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title('Original Image - grayscale')
    plt.show()

    # Performing histogram equalization
    im_eq, hist_orig, hist_eq = histogram_equalize(im)

    # Displaying the equalized image
    plt.imshow(im_eq, cmap='gray')
    plt.axis('off')
    plt.title('Equalized Image - grayscale')
    plt.show()

    # Performing optimal quantization
    im_quant, error = quantize(im, n_quant=3, n_iter=10)

    # Displaying the quantized image
    plt.imshow(im_quant, cmap='gray')
    plt.axis('off')
    plt.title('Quantized Image - grayscale')
    plt.show()


def rgb_example(filename):
    # Reading a grayscale image
    im = read_image(filename, representation=RGB)

    # Displaying the original image
    plt.imshow(im)
    plt.axis('off')
    plt.title('Original Image - RGB')
    plt.show()

    # Performing histogram equalization
    im_eq, hist_orig, hist_eq = histogram_equalize(im)

    # Displaying the equalized image
    plt.imshow(im_eq)
    plt.axis('off')
    plt.title('Equalized Image - RGB')
    plt.show()

    # Performing optimal quantization
    im_quant = quantize_rgb(im, n_quant=5)

    # Displaying the quantized image
    plt.imshow(im_quant)
    plt.axis('off')
    plt.title('Quantized Image - RGB')
    plt.show()


if __name__ == '__main__':
    grayscale_example("monkeys.jpeg")
    rgb_example("monkeys.jpeg")
