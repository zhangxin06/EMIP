import os
import torch
import argparse
import numpy as np
import yaml
from scipy import misc

import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from dataset.dataset import eval_dataset as EvalDataset
from model.EMIP_short.model import CoUpdater as Network


def evaluator(model, val_root, map_save_path, trainsize=352, data_name='MoCA'):
    val_loader = EvalDataset(images_root=val_root,
                             testsize=trainsize,
                             dataset_type=data_name)

    model.eval()
    with torch.no_grad():
        for i in range(val_loader.size):
            image1, image2, name, video_name, shape = val_loader.load_data()
            image1 = image1.cuda()
            image2 = image2.cuda()
            model = model.cuda()

            output = model(image1, image2)
            output = F.upsample(output[0], size=shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            map_save_path_final = map_save_path + video_name + '/'
            os.makedirs(map_save_path_final, exist_ok=True)
            from PIL import Image
            Image.fromarray(output*255).convert('L').save(map_save_path_final + name + '.png')
            print('>>> prediction save at: {}'.format(map_save_path + video_name + '/' + name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='/home/zhangxin/EMIP',
                        help='the path to save model and log')
    parser.add_argument('--snap_path', type=str, default='/home/zhangxin/EMIP/snapshots/log/Net_epoch_best.pth',
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='2',
                        help='train use gpu')
    parser.add_argument('--config', default="configs/configs.yaml")
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='train use gpu')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='train use gpu')

    opt = parser.parse_args()
    with open(opt.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    os.makedirs(opt.save_path, exist_ok=True)

    print('>>> configs:', opt)

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    cudnn.benchmark = True

    # build the model
    model = Network(args=config['model']['args'])
    checkpoint = torch.load(opt.snap_path)
    model_dict = model.state_dict()
    print('load model from ', opt.snap_path)
    if opt.multi_gpu:
        pretrained_dict = {k.split('module.')[-1]: v for k, v in checkpoint.items() if (k.split('module.')[-1] in model_dict)}
    else:
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    model.cuda()

    path_dic = {'CAD_eval': '/home/zhangxin/Datasets/VCOD/CAD/',
                'MoCA_test': '/home/zhangxin/Datasets/VCOD/MoCA_Video/TestDataset_per_sq/'}

    for data_name in ['CAD_eval', 'MoCA_test']:
        map_save_path = opt.save_path + "/{}/".format(data_name)
        val_root = path_dic[data_name]
        evaluator(
            model=model,
            val_root=val_root,
            map_save_path=map_save_path,
            trainsize=config['val_dataset']['inp_size'],
            data_name=data_name)
