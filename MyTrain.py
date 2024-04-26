import os
import logging
import subprocess

import numpy as np
from datetime import datetime
import random

import yaml
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torchvision.utils import make_grid
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import scipy.misc
from matplotlib import pyplot as plt

# customized libraries
import eval.metrics as Measure
from model.EMIP.model import CoUpdater as Network
from utils.utils import clip_gradient
from dataset.dataset import get_loader, test_dataset
from loss.loss_pred import hybrid_e_loss
from loss.loss_flow import unFlowLoss



def train(train_loader, model, optimizer, epoch, save_path, writer, config, opt):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0.0
    epoch_step = 0
    loss_flow_func = unFlowLoss()
    try:
        for i, (image1, image2, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            image1 = image1.cuda()
            image2 = image2.cuda()
            gts = gts.cuda()
            model = model.cuda()

            preds = model(image1, image2)
            loss_pred = hybrid_e_loss(preds[0], gts)

            image_pair = torch.cat((image1, image2), dim=1)
            flow_pair = []
            for idx in range(len(preds[1])):
                flow_pair.append(torch.cat((preds[1][idx], preds[2][idx]), dim=1))
            loss_flow = loss_flow_func.compute_loss(flow_pair, image_pair)
            loss = loss_pred + loss_flow[0]
            loss.backward()
            clip_gradient(optimizer, config['clip'])
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data
            if (opt.multi_gpu and opt.local_rank == 0) or opt.multi_gpu is False:
                if i % 20 == 0 or i == total_step or i == 1:
                    print(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f} loss_flow: {:0.4f}'.
                        format(datetime.now(), epoch, config['epoch_max'], i, total_step, loss.data, loss_pred.data,
                               loss_flow[0].data))
                    logging.info(
                        '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f} loss_flow: {:0.4f}'.
                        format(epoch, config['epoch_max'], i, total_step, loss.data, loss_pred.data, loss_flow[0].data))
                    writer.add_scalars('Loss_Statistics',
                                       {'loss_pred': loss_pred.data, 'loss_flow': loss_flow[0].data,
                                        'Loss_total': loss.data},
                                       global_step=step)
        loss_all /= epoch_step
        if (opt.multi_gpu and opt.local_rank == 0) or opt.multi_gpu is False:
            logging.info(
                '[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, config['epoch_max'], loss_all))
            writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if (opt.multi_gpu and opt.local_rank == 0) or opt.multi_gpu is False:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
            print('Save checkpoints successfully!')
            raise


def val(test_loader, model, epoch, save_path, writer, config, opt):
    """
    validation function
    """
    global best_metric_dict, best_mae, best_epoch, val_step
    wFm = Measure.WeightedFmeasure()
    Sm = Measure.Smeasure()
    MAE = Measure.MAE()
    metrics_dict = dict()

    model.eval()
    model.cuda()
    epoch_step_val = 0
    loss_all_val = 0
    if opt.multi_gpu is False or opt.local_rank == 0:
        pbar = tqdm(total=test_loader.size, leave=False, desc='val')
    else:
        pbar = None

    with torch.no_grad():
        for i in range(test_loader.size):
            image1, image2, gt, gt_tensor, _, _, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gts = gt_tensor.cuda()
            gts = gts.unsqueeze(0)
            image1 = image1.cuda()
            image2 = image2.cuda()

            res = model(image1, image2)

            loss_val_pred = hybrid_e_loss(res[0], gts)

            val_step += 1
            epoch_step_val += 1
            loss_all_val += loss_val_pred

            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            wFm.step(pred=res, gt=gt)
            Sm.step(pred=res, gt=gt)
            MAE.step(pred=res, gt=gt)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        metrics_dict.update(Sm=Sm.get_results()['sm'])
        metrics_dict.update(wFm=wFm.get_results()['wfm'])
        metrics_dict.update(MAE=MAE.get_results()['mae'])

        cur_mae = metrics_dict['MAE']
        loss_all_val /= epoch_step_val
        if (opt.multi_gpu and opt.local_rank == 0) or opt.multi_gpu is False:
            logging.info('[Val Info]: Epoch [{:03d}/{:03d}], Loss_val_epoch: {:.4f}'.format(epoch, config['epoch_max'],
                                                                                            loss_all_val))
            writer.add_scalar('Loss_val_epoch', loss_all_val, global_step=epoch)

            if epoch == 1:
                best_mae = cur_mae
                best_metric_dict = metrics_dict
                best_epoch = epoch
                print('[Cur Epoch: {}] Metrics (wFm={}, Sm={}, MAE={})'.format(
                    epoch, metrics_dict['wFm'], metrics_dict['Sm'], metrics_dict['MAE']))
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! the first epoch.')
                logging.info('[Cur Epoch: {}] Metrics (wFm={}, Sm={}, MAE={})'.format(
                    epoch, metrics_dict['wFm'], metrics_dict['Sm'], metrics_dict['MAE']))
            else:
                if cur_mae < best_mae:
                    best_metric_dict = metrics_dict
                    best_mae = cur_mae
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                    print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
                else:
                    print('>>> not find the best epoch -> continue training ...')
                print(
                    '[Cur Epoch: {}] Metrics (wFm={}, Sm={}, MAE={})\n[Best Epoch: {}] Metrics (wFm={}, Sm={}, MAE={})'.format(
                        epoch, metrics_dict['wFm'], metrics_dict['Sm'], metrics_dict['MAE'],
                        best_epoch, best_metric_dict['wFm'], best_metric_dict['Sm'], best_metric_dict['MAE']))
                logging.info(
                    '[Cur Epoch: {}] Metrics (wFm={}, Sm={}, MAE={})\n[Best Epoch:{}] Metrics (wFm={}, Sm={}, MAE={})'.format(
                        epoch, metrics_dict['wFm'], metrics_dict['Sm'], metrics_dict['MAE'],
                        best_epoch, best_metric_dict['wFm'], best_metric_dict['Sm'], best_metric_dict['MAE']))


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29501"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
        os.environ["MASTER_PORT"] = "29501"
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    local_rank = int(rank % num_gpus)
    device = torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, local_rank, device

def adp_lr(bs):
    base_bs = 36
    base_lr = 1e-4
    multiple = bs / base_bs
    new_lr = base_lr * pow(multiple, 0.5)
    return new_lr


def setup_seed(seed=3407):
    print('seed:{}'.format(seed))
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn. enabled = False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/configs.yaml")
    parser.add_argument('--num_workers', type=int, default=4,
                        help='train use gpu')
    parser.add_argument('--multi_gpu', type=bool, default=False,
                        help='train use gpu')
    parser.add_argument("--rank", type=int, default=-1, help="")
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--gpu_id", type=str, default='1', help="")
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--test_mode', default=False, help='test/train mode')
    opt = parser.parse_args()

    with open(opt.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    setup_seed(config['seed'])
    save_path = config['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    if opt.multi_gpu:
        rank, local_rank, device = setup_distributed()
        opt.rank = rank
        opt.local_rank = local_rank

        # build the model
        model = Network(args=config['model']['args'])
        model = model.to(device)

        # DistributedDataParallel
        model = DDP(model.cuda(), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if config['load']['path'] is not None:
            checkpoint = torch.load(config['load']['path'])
            model_dict = model.state_dict()
            print('load model from ', config['load']['path'])
            pretrained_dict = {'module.' + k: v for k, v in checkpoint.items() if 'module.' + k in model_dict}
            model_dict.update(pretrained_dict)

            if config['load']['flow_path'] is not None:
                checkpoint_flow = torch.load(config['load']['flow_path'])
                flow_dict = {'module.GMFlow.' + k: v for k, v in checkpoint_flow['model'].items() if
                             'module.GMFlow.' + k in model_dict}
                model_dict.update(flow_dict)

            model.load_state_dict(model_dict)
    else:
        # build the model
        model = Network(args=config['model']['args'])

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

        if config['load']['path'] is not None:
            checkpoint = torch.load(config['load']['path'])
            model_dict = model.state_dict()
            pretrained_dict_ori = {k: v for k, v in checkpoint.items() if ((k in model_dict and 'PromptInteract.PatchEmbed.proj.weight' not in k and "mask_downscaling" not
                                                                            in k)
                                or ('backbone.pvtv2_en' in k))}
            pretrained_dict = {}
            for k, v in pretrained_dict_ori.items():
                if 'backbone.pvtv2_en' in k:
                    key = k.replace('backbone.pvtv2_en', 'backbone.feat_net.pvtv2_en')
                    pretrained_dict[key] = v
                elif 'PromptInteract' in k:
                    pretrained_dict[k] = v
                    key = k.replace('PromptInteract', 'cod_adaptor_prompt')
                    pretrained_dict[key] = v
                else:
                    pretrained_dict[k] = v


            model_dict.update(pretrained_dict)
            if config['load']['flow_path'] is not None:
                checkpoint_flow = torch.load(config['load']['flow_path'])
                flow_dict = {'GMFlow.' + k: v for k, v in checkpoint_flow['model'].items() if
                             'GMFlow.' + k in model_dict}
                model_dict.update(flow_dict)
            model.load_state_dict(model_dict)


    for name, para in model.named_parameters():
        if "GMFlow" in name and 'dwconv' not in name and 'adaptor' not in name:
            para.requires_grad = False


    print('Now device id:...')
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    print(torch.cuda.current_device())

    # load data
    print('load data...')
    train_loader = get_loader(image_root=config['train_dataset']['image_path'],
                              gt_root=config['train_dataset']['gt_path'],
                              batchsize=config['train_dataset']['batch_size'],
                              trainsize=config['train_dataset']['inp_size'],
                              num_workers=opt.num_workers,
                              pin_memory=False,
                              multi_gpu=opt.multi_gpu,
                              dataset_type=config['train_dataset']['dataset_type'])
    val_loader = test_dataset(images_root=config['val_dataset']['image_path'],
                              gts_root=config['val_dataset']['gt_path'],
                              testsize=config['val_dataset']['inp_size'],
                              dataset_type=config['val_dataset']['dataset_type'])
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + '4090_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    val_step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_epoch = 0
    best_mae = 1

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), config['optimizer']['lr'], weight_decay=config['optimizer']['weight_decay'])
    schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config['epoch_max'], eta_min=config['lr_min'])

    print(">>> start train...")
    for epoch in range(1, config['epoch']):
        # schedule
        schedule.step()

        writer.add_scalar('learning_rate_base', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        logging.info('>>> current lr_base: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        # train
        train(train_loader, model, optimizer, epoch, save_path, writer, config, opt)
        if epoch > 0:
            # validation
            val(val_loader, model, epoch, save_path, writer, config, opt)


