"""
---------------------------- -------------------BASA"D
@Author: Nitay880
------------------------------------------------------------------------
"""

# libraries - numpy,imageio,skimage,tensorflow,keras,matplotlib
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Input, Conv2D, Activation
from sklearn.model_selection import train_test_split
from scipy.ndimage.filters import convolve
from tensorflow.keras.optimizers import Adam
import utill_functions
import matplotlib.pyplot as plt

# ========================::CONSTANTAS::===================#
# =============::DENOISING CONFIGURATION::========================#
DENOISING_NUM_CHANNELS = 48
DENOISING_QUICK_VALIDATION_SIZE = 30
DENOISING_QUICK_NUM_EPOCHS = 2
DENOISING_QUICK_STEPS_PER_EPOCH = 3
DENOISING_QUICK_BATCH_SIZE = 10
DENOISING_VALIDATION_SIZE = 1000
DENOISING_NUM_EPOCHS = 5
DENOISING_STEPS_PER_EPOCH = 100
DENOISING_BATCH_SIZE = 100
CROP_SIZE_FOR_DENOISING = (24, 24)
# =============::DEBLURRING CONFIGURATION::========================#
DEBLURRING_NUM_CHANNELS = 32
DEBLURRING_QUICK_VALIDATION_SIZE = 30
DEBLURRING_QUICK_NUM_EPOCHS = 2
DEBLURRING_QUICK_STEPS_PER_EPOCH = 3
DEBLURRING_QUICK_BATCH_SIZE = 10
DEBLURRING_VALIDATION_SIZE = 1000
DEBLURRING_NUM_EPOCHS = 10
DEBLURRING_STEPS_PER_EPOCH = 100
DEBLURRING_BATCH_SIZE = 100
BLURRING_KERNEL_SIZES = [7]
CROP_SIZE_FOR_DEBLURRING = (16, 16)
# ==========::GENERAL CONSTANTAS::=======#
RGB = 2
GRAY_SCALE = 1
MAX_SCALE_VALUE = 255
VALID_INPUT = 1
ERROR = 0


# ==========::END CONSTANTAS::=======#

# ==========::Utils functions::=======#
def add_gaussian_noise(image, min_sigma, max_sigma):
    """

    :param image:
    :param min_sigma:
    :param max_sigma:
    :return:
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    noisy = image + noise
    noisy = nearest_fraction(noisy, 1 / 255)
    return np.clip(noisy, 0, 1)


def add_motion_blur(image, kernel_size, angle):
    """
    The following function adds motion blur to a given image
    :param image: 2d numpy array
    :param kernel_size: convolved kernel size
    :param angle: angle of the blur-line
    :return: blurred image, clipped to [0,1]
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    convolved = convolve(image, kernel)
    return np.clip(nearest_fraction(convolved, 1 / 255), 0, 1)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    The following function adds a random motion blur to a given
    image, random in terms of - kernel and angle.
    :param image:2d numpy array.
    :param list_of_kernel_sizes: list of kernel sizes to choose randomly from.
    :return: blurred image, clipped to [0,1] numpy 2d array.
    """
    n = len(list_of_kernel_sizes)
    kernel_index = 0 if n == 1 else np.random.randint(0, n - 1)
    angle = np.random.uniform(0, np.pi)
    return add_motion_blur(image, list_of_kernel_sizes[kernel_index], angle)

def nearest_fraction(im, frac):
    """
    The following function reterns the nearst frac fraction to each coordinate of image.
    :param im: input image
    :param frac: fraction <=1
    :return: given frac= 1/j , this function returns round(im/(1/j))*1/j
    """
    im2 = np.array(im)
    im2 /= frac
    im2 = np.round(im2)
    im2 *= frac
    return im2


def read_image(filename, representation):
    """
    Read Image

    the following function takes a file name of
    an image, and a given output-representation'
    and returns the image represented as a matrix
    in the given output-representation.
    :param filename:  String ( Not Null)
    :param representation:  0 for Grayscale, 1 for RGB, not null
    :return: (width,height,) OR (width,height,3) float64 numpy array.
    """
    image = imread(filename)

    # if it's an rgb image, it will turn to grayscale as required, else it will stay in grayscale
    if representation == GRAY_SCALE and len(image.shape) > 2:
        # len(image.shape)>2 means that it has more then one value per pixel, in that case it's rgb...
        image = rgb2gray(image)
    """
    The case where rep=2 is redundant since if the input is rgb it will stay rgb,
    and in other case where our image is gray scale, there is no option that rep=2 as
    we can assume that from the instructions of this exercise...
    """
    image_float = image.astype(np.float64)  # float conversion
    if np.max(image_float) > 1:  # case it needs normalization
        image_float /= MAX_SCALE_VALUE  # normalize intensities
    return image_float


def sample_patch(image, crop_size, im2=None):
    """
    The following function generates a random patch in size crop_size
    and samples it from image, and im2 (if exists)
    returns both patchs, in this order (im_patch,im2_patch)
    :param image:2d numpy array
    :param crop_size:2-tuple (height,width)
    :param im2: 2d numpy array
    :return: (im_patch,im2_patch (optional) )
    """
    patch_h, patch_w = crop_size
    top_left_patch_x, top_left_patch_y = np.random.randint(0, image.shape[1] - patch_w), np.random.randint(0,
                                                                                                           image.shape[
                                                                                                               0] - patch_h)
    patch = image[top_left_patch_y:top_left_patch_y + patch_h, top_left_patch_x:top_left_patch_x + patch_w]
    if im2 is not None:
        patch2 = im2[top_left_patch_y:top_left_patch_y + patch_h, top_left_patch_x:top_left_patch_x + patch_w]
        return patch, patch2
    else:
        return patch

# ==========::END Utils functions::=======#

def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    The following function generates a
    random data out of filenames,
    it keeps a cache dictionary of images,
    once it is called, it generates
    2 patches, out of the random image,
    one is sampled from image-0.5, the other
    is sampled from blurred_image-0.5.

    :param filenames: List of image files
    :param batch_size: The size of the batch that returned in each call
    :param corruption_func: corruption function (has to work on 2d numpy arrays)
    :param crop_size:A tuple (height, width) specifying the crop size of the patches to extract
    :return: the function returns two batches of size batch size, one is related to the blurred image (blurred by corruption func),
             the other one is related to the original one.
    """
    # start with an empty dict
    image_dict = {}
    crop_height, crop_width = crop_size
    larger_crop = (crop_height * 3, crop_width * 3)
    while True:
        indexes = np.random.choice(range(len(filenames)), batch_size)
        source_batch = np.zeros((batch_size, crop_height, crop_width, 1))
        target_batch = np.zeros((batch_size, crop_height, crop_width, 1))
        i = 0
        for index in indexes:
            image_name = filenames[index]
            if image_name not in image_dict.keys():
                # add image to the cache
                image_dict[image_name] = read_image(image_name, GRAY_SCALE)
            image = image_dict[image_name]
            larger_patch = sample_patch(image, larger_crop)
            larger_patch_corrupted = corruption_func(larger_patch)
            regular_patch, corrupted_patch = sample_patch(larger_patch, crop_size, larger_patch_corrupted)
            source_batch[i] = corrupted_patch[:, :, np.newaxis] - 0.5
            target_batch[i] = regular_patch[:, :, np.newaxis] - 0.5
            i += 1
        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    """
    The following function builds a res block that it's input is
    in the following way
    inp--->conv2d-->relu-->conv2d--->add--->relu
    :param input_tensor: input
    :param num_channels: number of channels in the conv kernel
    :return: resblock which it's input is input_tensor with num_channels output channels.
    """
    inp = input_tensor
    b = Conv2D(num_channels, (3, 3), padding='same')(inp)
    b = Activation('relu')(b)
    b = Conv2D(num_channels, (3, 3), padding='same')(b)
    b = Add()([inp, b])
    b = Activation('relu')(b)
    return b


def build_nn_model(height, width, num_channels, num_res_blocks):
    """

    :param height:
    :param width:
    :param num_channels:
    :param num_res_blocks:
    :return:
    """
    inp = Input(shape=(height, width, 1))
    b = Conv2D(num_channels, (3, 3), padding='same')(inp)
    b = Activation('relu')(b)
    while num_res_blocks > 0:
        b = resblock(b, num_channels)
        num_res_blocks -= 1
    b = Conv2D(1, (3, 3), padding='same')(b)
    b = Add()([inp, b])
    model = Model(inputs=inp, outputs=b)
    return model


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Parameters:
    model – a general neural network model for image restoration.
    images – a list of file paths pointing to image files. You should assume these paths are complete, and
    should append anything to them.
    corruption_func – same as described in section 3.
    batch_size – the size of the batch of examples for each iteration of SGD.
    steps_per_epoch – The number of update steps in each epoch.
    num_epochs – The number of epochs for which the optimization will run.
    :return:None
    """
    model_shape = model.input_shape
    CROP_SIZE = (model_shape[1], model_shape[2])
    train_data, test_data = train_test_split(images, test_size=0.2)
    train_generator = load_dataset(train_data, batch_size, corruption_func, CROP_SIZE)
    test_generator = load_dataset(test_data, batch_size, corruption_func, CROP_SIZE)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs,
                        validation_data=test_generator, use_multiprocessing=True, validation_steps=num_valid_samples)


def restore_image(corrupted_image, base_model):
    """
    The following function gets a corrupted image
    and fixes it according to the input base model
    :param corrupted_image: 2d numpy array
    :param base_model: Keras NN model
    :return: Fixed image 2d numpy array
    """
    a = Input(shape=(corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    new_model = Model(inputs=a, outputs=b)
    corrupted_image -= 0.5
    prediction = new_model.predict(corrupted_image[np.newaxis, :, :, np.newaxis])[0]
    new_im = prediction.reshape(corrupted_image.shape[0], corrupted_image.shape[1])
    new_im += 0.5
    new_im = np.clip(new_im, 0, 1)
    return new_im.astype(np.float64)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    The following function trains denoising model
    :param num_res_blocks: number of residual blocks (default is 5)
    :param quick_mode: Boolean that indicates the function to do a quick training
    :return: Keras trained denoising model
    """
    gauss = lambda x: add_gaussian_noise(x, 0, 0.2)
    images = sol5_utils.images_for_denoising()
    model = build_nn_model(CROP_SIZE_FOR_DENOISING[0], CROP_SIZE_FOR_DENOISING[1], DENOISING_NUM_CHANNELS,
                           num_res_blocks)
    if quick_mode:
        train_model(model, images, gauss, DENOISING_QUICK_BATCH_SIZE, DENOISING_QUICK_STEPS_PER_EPOCH,
                    DENOISING_QUICK_NUM_EPOCHS,
                    DENOISING_QUICK_VALIDATION_SIZE)
    else:
        train_model(model, images, gauss, DENOISING_BATCH_SIZE, DENOISING_STEPS_PER_EPOCH, DENOISING_NUM_EPOCHS,
                    DENOISING_VALIDATION_SIZE)

    return model


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    The following function trains a deblurring model.
    :param num_res_blocks: number of residual blocks (default is 5)
    :param quick_mode: Boolean that indicates the function to do a quick training
    :return: Keras trained deblurring model
    """
    blurr = lambda x: random_motion_blur(x, BLURRING_KERNEL_SIZES)
    images = sol5_utils.images_for_deblurring()
    model = build_nn_model(CROP_SIZE_FOR_DEBLURRING[0], CROP_SIZE_FOR_DEBLURRING[1], DEBLURRING_NUM_CHANNELS,
                           num_res_blocks)
    if quick_mode:
        train_model(model, images, blurr, DEBLURRING_QUICK_BATCH_SIZE, DEBLURRING_QUICK_STEPS_PER_EPOCH,
                    DEBLURRING_QUICK_NUM_EPOCHS,
                    DEBLURRING_QUICK_VALIDATION_SIZE)
    else:
        train_model(model, images, blurr, DEBLURRING_BATCH_SIZE, DEBLURRING_STEPS_PER_EPOCH, DEBLURRING_NUM_EPOCHS,
                    DEBLURRING_VALIDATION_SIZE)
    return model


def deep_prior_restore_image(corrupted_image):
    """
    The following function implements deep prior image restoration
    :param corrupted_image: 2d numpy array that represents a corrupted image.
    :return: restored image (2d numpy array)
    """
    num_res_blocks = 5
    num_channels = 28
    num_it = 700


    image_height, image_width = corrupted_image.shape
    model = build_nn_model(image_height, image_width, num_channels, num_res_blocks)
    model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
    #uniform noise sampled in the range [0,0.1]
    random_noise = np.random.uniform(0, 0.1, (image_height, image_width))
    plt.imshow(corrupted_image, cmap=plt.gray())
    plt.show()

    for i in range(num_it):
        model.fit(x=random_noise[np.newaxis, :, :, np.newaxis], y=corrupted_image[np.newaxis, :, :, np.newaxis],
                  epochs=1)
        print(num_it - i)
        target = random_noise
        prediction = model.predict(target[np.newaxis, :, :, np.newaxis])[0]
        new_im = prediction.reshape(target.shape[0], target.shape[1])
    return new_im