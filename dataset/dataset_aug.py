import os
import numpy as np
import torch
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
from dataset.data_augment import randomRotation, colorEnhance, randomPeper, random_flip_horizontal, random_flip_vertical


# dataset for training
class ObjDataset(data.Dataset):
    def __init__(self, images_root, gts_root, trainsize, dataset='MoCA'):
        self.trainsize = trainsize

        # get filenames
        ori_root = images_root
        self.images = []
        self.gts = []
        self.flows = []
        self.image_pairs_list = []
        self.extra_info = []

        if dataset == 'MoCA':
            for video_name in os.listdir(ori_root):
                image_root = images_root + video_name + '/Imgs/'
                gt_root = gts_root + video_name + '/GT/'
                self.image = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
                self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

                # sorted files
                self.image = sorted(self.image)
                self.gt = sorted(self.gt)[0:-1]

                self.gts += self.gt

                for i in range(len(self.image) - 1):
                    self.image_pairs_list += [[self.image[i], self.image[i + 1]]]
                    frame_name = self.image[i].split('/')[-1].split('.')[0]
                    self.extra_info += [(video_name, frame_name)]

                assert len(self.image_pairs_list) == len(self.gts)
        elif dataset == 'VSOD':
            self.images_all = [f for f in os.listdir(images_root) if f.endswith('.jpg')]
            self.images = sorted(self.images)

            for idx, image_nm in self.images_all:
                if idx==0: continue
                if self.images_all[idx].split('_')[0] == self.images_all[idx-1].split('_')[0]:
                    self.images += [[images_root + self.images_all[idx-1], images_root + self.images_all[idx]]]
                    self.gts += (gts_root + self.images_all[idx - 1]).replace('.jpg', '.png')
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
        assert self.image_pairs_list[index][0].split('/')[-1].split('.')[0] == self.gts[index].split('/')[-1].split('.')[0]
        image1 = self.rgb_loader(self.image_pairs_list[index][0])
        image2 = self.rgb_loader(self.image_pairs_list[index][1])
        gt = self.binary_loader(self.gts[index])

        # data augumentation
        image1, image2, gt = randomRotation(image1, image2, gt)
        image1, image2, gt = random_flip_horizontal(image1, image2, gt)
        image1, image2, gt = random_flip_vertical(image1, image2, gt)
        image1 = colorEnhance(image1)
        image2 = colorEnhance(image2)
        gt = randomPeper(gt)

        image1 = self.img_transform(image1)
        image2 = self.img_transform(image2)
        gt = self.gt_transform(gt)

        return image1, image2, gt

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
def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True, multi_gpu=False, dataset_type='MoCA'):
    dataset = ObjDataset(image_root, gt_root, trainsize, dataset_type)
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
    def __init__(self, images_root, gts_root, testsize, dataset_type='MoCA'):
        self.testsize = testsize

        self.gt_list = []
        self.image_pairs_list = []
        self.extra_info = []
        ori_root = images_root
        for video_name in os.listdir(ori_root):
            if 'CAD' in dataset_type:
                image_root = images_root + video_name + '/frames/'
            else:
                image_root = images_root + video_name + '/Imgs/'
            gt_root = gts_root + video_name + '/GT/'
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

            self.images = sorted(self.images)
            self.gts = sorted(self.gts)[0:-1]
            self.gt_list += self.gts

            for i in range(len(self.images) - 1):
                self.image_pairs_list += [[self.images[i], self.images[i + 1]]]
                frame_name = self.images[i].split('/')[-1].split('.')[0]
                self.extra_info += [(video_name, frame_name)]

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
        assert self.image_pairs_list[self.index][0].split('/')[-1].split('.')[0] == self.gt_list[self.index].split('/')[-1].split('.')[0]
        image1 = self.rgb_loader(self.image_pairs_list[self.index][0])
        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.rgb_loader(self.image_pairs_list[self.index][1])
        image2 = self.transform(image2).unsqueeze(0)

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

        return image1, image2, gt, gt_tensor, name, video_name, np.array(image_for_post)

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


class eval_dataset:
    def __init__(self, images_root, testsize, dataset_type='MoCA'):
        self.testsize = testsize

        self.gt_list = []
        self.image_pairs_list = []
        self.extra_info = []
        ori_root = images_root
        for video_name in os.listdir(ori_root):
            if 'CAD' in dataset_type:
                image_root = images_root + video_name + '/frames/'
            else:
                image_root = images_root + video_name + '/Imgs/'
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.images = sorted(self.images)

            for i in range(len(self.images) - 1):
                self.image_pairs_list += [[self.images[i], self.images[i + 1]]]
                frame_name = self.images[i].split('/')[-1].split('.')[0]
                self.extra_info += [(video_name, frame_name)]

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.image_pairs_list)
        self.index = 0

    def load_data(self):
        image1 = self.rgb_loader(self.image_pairs_list[self.index][0])
        shape = (image1.height, image1.width)
        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.rgb_loader(self.image_pairs_list[self.index][1])
        image2 = self.transform(image2).unsqueeze(0)

        video_name = self.extra_info[self.index][0]
        name = self.extra_info[self.index][1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image1, image2, name, video_name, shape

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