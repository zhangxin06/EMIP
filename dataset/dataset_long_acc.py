import os
import numpy as np
import torch
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
from dataset.data_augment import randomRotation, colorEnhance, randomPeper


# dataset for training
class ObjDataset(data.Dataset):
    def __init__(self, images_root, gts_root, trainsize, dataset='MoCA'):
        self.trainsize = trainsize

        # get filenames
        ori_root = images_root
        self.image_pairs_list = []
        self.extra_info = []
        self.num_frames = {}
        self.num_gts = {}
        self.shape = {}
        self.videos = []

        if dataset == 'MoCA':
            for video_name in os.listdir(ori_root):
                image_root = images_root + video_name + '/Imgs/'
                gt_root = gts_root + video_name + '/GT/'
                self.videos.append(video_name)
                self.image = [image_root + f for f in os.listdir(image_root) if
                              f.endswith('.jpg') or f.endswith('.png')]
                self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
                # sorted files
                self.image = sorted(self.image)
                self.gt = sorted(self.gt)
                self.num_frames[video_name] = self.image
                self.num_gts[video_name] = self.gt

                _mask = np.array(Image.open(self.gt[0]).convert("P"))
                self.shape[video_name] = np.shape(_mask)  # {'blackswan': (480, 854)}

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.num_frames)
        print('>>> trainig/validing with {} videos'.format(self.size))

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}  # {'name': 'blackswan', 'num_frames': 50, 'shape': (480, 854)}
        info['name'] = video
        info['num_frames'] = len(self.num_frames[video])
        info['shape'] = self.shape[video]

        N_frames = np.empty((info['num_frames'],) + (3, 352, 352), dtype=np.float32)  # (50 480 854 3)
        N_masks = np.empty((info['num_frames'],) + (1, 352, 352), dtype=np.float32)  # (50 480 854)
        for f in range(info['num_frames']):
            img_file = self.num_frames[video][f]
            gt_file = self.num_gts[video][f]
            image = self.rgb_loader(img_file)
            gt = self.binary_loader(gt_file)
            N_frames[f] = self.img_transform(image)
            N_masks[f] = self.gt_transform(gt)

        return N_frames, N_masks, info

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
class test_dataset(data.Dataset):
    def __init__(self, images_root, gts_root, testsize, dataset='MoCA'):
        self.testsize = testsize

        # get filenames
        ori_root = images_root
        self.image_pairs_list = []
        self.extra_info = []
        self.num_frames = {}
        self.num_gts = {}
        self.shape = {}
        self.videos = []

        for video_name in os.listdir(ori_root):
            if 'CAD' in ori_root:
                image_root = images_root + video_name + '/Imgs/'  # TODO
            else:
                image_root = images_root + video_name + '/Imgs/'
            gt_root = gts_root + video_name + '/GT/'
            self.videos.append(video_name)
            self.image = [image_root + f for f in os.listdir(image_root) if
                          f.endswith('.jpg') or f.endswith('.png')]
            self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
            # sorted files
            self.image = sorted(self.image)
            self.gt = sorted(self.gt)
            self.num_frames[video_name] = self.image
            self.num_gts[video_name] = self.gt

            _mask = np.array(Image.open(self.gt[0]).convert("P"))
            self.shape[video_name] = np.shape(_mask)  # {'blackswan': (480, 854)}

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.num_frames)
        self.index = 0
        print('>>> validing with {} videos'.format(self.size))

    def load_data(self):
        video = self.videos[self.index]
        info = {}  # {'name': 'blackswan', 'num_frames': 50, 'shape': (480, 854)}
        info['name'] = video
        info['num_frames'] = len(self.num_frames[video])
        info['shape'] = self.shape[video]
        info['frames_name'] = [item.split('/')[-1].replace('.jpg', '') for item in self.num_frames[video]]

        N_frames = np.empty((info['num_frames'],) + (3, 352, 352), dtype=np.float32)  # (50 480 854 3)
        N_masks = np.empty((info['num_frames'],) + (1, 352, 352), dtype=np.float32)  # (50 480 854)
        N_gts = np.empty((info['num_frames'],) + info['shape'], dtype=np.float32)
        for f in range(info['num_frames']):
            img_file = self.num_frames[video][f]
            gt_file = self.num_gts[video][f]
            image = self.rgb_loader(img_file)
            gt = self.binary_loader(gt_file)
            N_gts[f] = gt
            N_frames[f] = self.img_transform(image)
            N_masks[f] = self.gt_transform(gt)

        self.index += 1
        self.index = self.index % self.size

        N_frames = torch.tensor(N_frames)
        N_masks = torch.tensor(N_masks)
        return N_frames, N_masks, N_gts, info

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


class eval_dataset(data.Dataset):
    def __init__(self, images_root, gts_root, testsize, dataset='MoCA'):
        self.testsize = testsize

        # get filenames
        ori_root = images_root
        self.image_pairs_list = []
        self.extra_info = []
        self.num_frames = {}
        self.shape = {}
        self.videos = []

        for video_name in os.listdir(ori_root):
            if 'CAD' in ori_root:
                image_root = images_root + video_name + '/frames/'
            else:
                image_root = images_root + video_name + '/Imgs/'
            gt_root = gts_root + video_name + '/GT/'
            self.videos.append(video_name)
            self.image = [image_root + f for f in os.listdir(image_root) if
                          f.endswith('.jpg') or f.endswith('.png')]
            self.gt = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]

            # sorted files
            self.image = sorted(self.image)
            self.num_frames[video_name] = self.image

            _mask = np.array(Image.open(self.gt[0]).convert("P"))
            self.shape[video_name] = np.shape(_mask)  # {'blackswan': (480, 854)}

        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # get size of dataset
        self.size = len(self.num_frames)
        self.index = 0
        print('>>> validing with {} videos'.format(self.size))

    def load_data(self):
        video = self.videos[self.index]
        info = {}  # {'name': 'blackswan', 'num_frames': 50, 'shape': (480, 854)}
        info['name'] = video
        info['num_frames'] = len(self.num_frames[video])
        info['shape'] = self.shape[video]
        info['frames_name'] = [item.split('/')[-1].replace('.jpg', '') for item in self.num_frames[video]]

        N_frames = np.empty((info['num_frames'],) + (3, 352, 352), dtype=np.float32)  # (50 480 854 3)
        N_masks = np.empty((info['num_frames'],) + (1, 352, 352), dtype=np.float32)  # (50 480 854)
        N_gts = np.empty((info['num_frames'],) + info['shape'], dtype=np.float32)
        for f in range(info['num_frames']):
            img_file = self.num_frames[video][f]
            image = self.rgb_loader(img_file)
            N_frames[f] = self.img_transform(image)

        self.index += 1
        self.index = self.index % self.size

        N_frames = torch.tensor(N_frames)
        N_masks = torch.tensor(N_masks)
        return N_frames, N_masks, N_gts, info

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