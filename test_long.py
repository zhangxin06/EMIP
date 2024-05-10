import os
import torch
import argparse
import numpy as np
import yaml
from scipy import misc

import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from dataset.dataset_long_acc import eval_dataset as EvalDataset
from model.EMIP_long.model_long import Model_long as Network


def evaluator(model, val_root, map_save_path, trainsize=352, data_name='MoCA'):
    val_loader = EvalDataset(images_root=val_root,
                             gts_root=val_root,
                             testsize=trainsize)

    model.eval()
    with torch.no_grad():
        for i in range(val_loader.size):
            N_frames, N_masks, N_gts, info = val_loader.load_data()
            N_frames = N_frames.squeeze(0).cuda()
            N_masks = N_masks.squeeze(0).cuda()
            model = model.cuda()
            frames_name = info['frames_name']

            memory_k = None
            memory_v = None
            for index in range(len(N_frames)):
                if index == 0:
                    preds, memory_k, memory_v = model(N_frames[index], N_frames[index + 1], index, memory_k, memory_v)
                else:
                    preds, memory_k, memory_v = model(N_frames[index - 1], N_frames[index], index, memory_k, memory_v)
                    memory_k = memory_k.detach()
                    memory_v = memory_v.detach()

                res = F.upsample(preds, size=info['shape'], mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                map_save_path_final = map_save_path + info['name'] + '/'
                os.makedirs(map_save_path_final, exist_ok=True)
                from PIL import Image
                Image.fromarray(res*255).convert('L').save(map_save_path_final + frames_name[index] + '.png')
                print('>>> prediction save at: {}'.format(map_save_path + info['name'] + '/' + frames_name[index]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path', type=str, default='E:/Projects/EMIP_long/result/',
                        help='the path to save model and log')
    parser.add_argument('--snap_path', type=str, default='E:/Projects/EMIP_long/snapshots/log/Net_epoch_best.pth',
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='0',
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

    path_dic = {'CAD_eval': '/home/zhangxin/dataset/Video/CAD/',
                'MoCA_test': '/home/zhangxin/dataset/Video/MoCA_test/',
                'moca_train': '/home/zhangxin/dataset/Video/MoCA_train/',
                'moca_pseudo': '/home/zhangxin/dataset/Video/MoCA_pseudo_test/',
                'test': '/home/zhangxin/data/dataset/CamouflagedAnimalDataset/CAD_test/',
                'cad-full': '/home/zhangxin/dataset/Video/CAD/',
                'DAVSOD': '/home/zhangxin/GoogleDownload/TestSet/DAVSOD/'}

    for data_name in ['CAD_eval']:  # 'CAMO', 'COD10K', 'NC4K' 'cad', 'moca'
        map_save_path = opt.save_path + "/{}/".format(data_name)
        val_root = path_dic[data_name]
        evaluator(
            model=model,
            val_root=val_root,
            map_save_path=map_save_path,
            trainsize=config['val_dataset']['inp_size'],
            data_name=data_name)
