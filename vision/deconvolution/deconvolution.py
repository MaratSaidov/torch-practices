import numpy as np
import scipy
import scipy.fft as fft
import scipy.stats as stats


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    mean = np.array([float(size // 2), float(size // 2)])
    cov = (sigma ** 2) * np.eye(2)

    x = np.arange(size)
    kernel = np.transpose([np.tile(x, size), np.repeat(x, size)]).reshape((size, size, 2))
    kernel = kernel.astype(np.float64)

    gaussian = stats.multivariate_normal.pdf(kernel, mean=mean, cov=cov)

    return gaussian / gaussian.sum()


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    return fft.fft2(h, shape)

THRESHOLD = 1e-10

def invert(elem):
    global THRESHOLD
    if abs(elem) <= THRESHOLD:
        return 0.
    return 1. / elem


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    global THRESHOLD
    THRESHOLD = threshold
    return np.vectorize(invert)(H)


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """

    H = fourier_transform(h, blurred_img.shape)
    H_inv = inverse_kernel(H, threshold=threshold)

    F_tilda = fft.fft2(blurred_img) * H_inv
    f_tilda = fft.ifft2(F_tilda, blurred_img.shape)
    reconstructed = abs(f_tilda)
    return reconstructed


def wiener_filtering(blurred_img, h, K=5e-4):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """

    H = fourier_transform(h, blurred_img.shape)
    H_conj = np.conjugate(H)
    denom = H_conj * H
    denom += K

    wiener = (H_conj / denom) * fft.fft2(blurred_img)

    f_tilda = fft.ifft2(wiener, blurred_img.shape)

    return np.absolute(f_tilda)


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    img_pred, img_gt = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img_pred - img_gt) ** 2)
    epsilon = 1e-5
    if mse == 0:
        raise ValueError("PSNR is uncountable for the same image.")
    psnr = 20.0 * np.log10(255.0 / (np.sqrt(mse) + epsilon))
    return psnr
