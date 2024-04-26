import os
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance

import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

def randomRotation(img1, img2, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img1 = img1.rotate(random_angle, mode)
        img2 = img2.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return img1, img2, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)

def random_flip_horizontal(img1, img2, label):
    # flip_horizontal
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img1, img2, label


def random_flip_vertical(img1, img2, label):
    # flip_vertical
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
        img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
    return img1, img2, label


def centerCrop(img1, img2, label, crop_size):
    img1 = F.center_crop(img1, crop_size)
    img2 = F.center_crop(img2, crop_size)
    label = F.center_crop(label, crop_size)
    return img1, img2, label


def randomCrop(img1, img2, label):
    border = 30
    image_width = img1.size[0]
    image_height = img1.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return img1.crop(random_region), img2.crop(random_region), label.crop(random_region)
