import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

import torch.utils.data as data
import torchvision.transforms as transforms
from dataset import flow_viz


def cv_random_flip(img1, img2, label, flow):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        flo = flow_viz.flow_to_image(flow)
        flow = np.fliplr(flow)
        flo = flow_viz.flow_to_image(flow)

    return img1, img2, label, flow


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


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


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


# dataset for training
class ObjDataset(data.Dataset):
    def __init__(self, images_root, gts_root, flows_root, trainsize, dataset='MoCA'):
        self.trainsize = trainsize

        # spatial augmentation params
        # self.crop_size = crop_size
        self.min_scale = 0.2
        self.max_scale = 0.5
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # get filenames
        ori_root = images_root
        self.images = []
        self.gts = []
        self.flows = []
        self.image_pairs_list = []
        self.extra_info = []

        self.datatype = dataset

        if dataset == 'MoCA':
            for video_name in os.listdir(ori_root):
                image_root = images_root + video_name + '/Imgs/'
                gt_root = gts_root + video_name + '/GT/'
                flow_root = flows_root + video_name + '/'
                self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
                self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

                # sorted files
                self.image = sorted(self.image)
                self.gt = sorted(self.gt)[0:-1]

                self.gts += self.gt
                self.flow = [flow_root + f for f in os.listdir(flow_root) if f.endswith('.jpg')]
                self.flows += sorted(self.flow)
                for i in range(len(self.image) - 1):
                    self.image_pairs_list += [[self.image[i], self.image[i + 1]]]
                    frame_name = self.image[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

                assert len(self.flows) == len(self.image_pairs_list) == len(self.gts)

            # filter mathcing degrees of files
        # self.filter_files() #
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.image_pairs_list)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        # read assest/gts/grads/depths
        assert self.image_pairs_list[index][0].split('/')[-1].split('.')[0] == self.gts[index].split('/')[-1].split('.')[0] == self.flows[index].split('/')[-1].split('.')[0]
        image1 = self.rgb_loader(self.image_pairs_list[index][0])
        image2 = self.rgb_loader(self.image_pairs_list[index][1])
        gt = self.binary_loader(self.gts[index])
        flow = self.rgb_loader(self.flows[index])

        # data augumentation
        """
        image1, image2, gt, flow = cv_random_flip(image1, image2, gt, flow)
        image1, image2, gt = randomCrop(image1, image2, gt)
        image1, image2, gt = randomRotation(image1, image2, gt)"""

        image1 = colorEnhance(image1)
        image2 = colorEnhance(image2)
        gt = randomPeper(gt)

        image1 = self.img_transform(image1)
        image2 = self.img_transform(image2)
        gt = self.gt_transform(gt)
        flow = self.img_transform(flow)

        return image1, image2, gt, flow

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# dataloader for training
def get_loader(image_root, gt_root, flow_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True, multi_gpu=False):
    dataset = ObjDataset(image_root, gt_root, flow_root, trainsize)
    print(dataset.__len__)
    if multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True
        )
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory,
                                      sampler=train_sampler)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batchsize,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, images_root, gts_root, flows_root, testsize):
        self.testsize = testsize

        self.gt_list = []
        self.image_pairs_list = []
        self.extra_info = []
        self.flows = []
        ori_root = images_root
        for video_name in os.listdir(ori_root):
            image_root = images_root + video_name + '/Imgs/'
            gt_root = gts_root + video_name + '/GT/'
            flow_root = flows_root + video_name + '/'
            self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

            self.image = sorted(self.image)
            self.gt = sorted(self.gt)[0:-1]
            self.gt_list += self.gt

            self.flow = [flow_root + f for f in os.listdir(flow_root) if f.endswith('.jpg')]
            self.flows += sorted(self.flow)
            for i in range(len(self.image) - 1):
                self.image_pairs_list += [[self.image[i], self.image[i + 1]]]
                frame_name = self.image[i].split('/')[-1].split('.')[0]
                self.extra_info += [(video_name, frame_name)]

            assert len(self.flows) == len(self.image_pairs_list) == len(self.gt_list)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.gt_transform_2 = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.image_pairs_list)
        self.index = 0

    def load_data(self):
        image1 = self.rgb_loader(self.image_pairs_list[self.index][0])
        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.rgb_loader(self.image_pairs_list[self.index][1])
        image2 = self.transform(image2).unsqueeze(0)
        flow = self.rgb_loader(self.flows[self.index])

        gt = self.binary_loader(self.gt_list[self.index])
        gt_tensor = self.gt_transform_2(gt)

        video_name = self.extra_info[self.index][0]
        name = self.extra_info[self.index][1]

        image_for_post = self.rgb_loader(self.image_pairs_list[self.index][0])
        image_for_post = image_for_post.resize(gt.size)

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size
        flow = self.transform(flow).unsqueeze(0)

        return image1, image2, gt, gt_tensor, flow,  name, video_name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def flow_loader(self, path):
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2)).astype(np.float32)

    def __len__(self):
        return self.size
