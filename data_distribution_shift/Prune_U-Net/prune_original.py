import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np

import os
import os.path as osp
import json
from optparse import OptionParser
from prune_utils_original import Pruner
from tqdm import tqdm
from finetune import finetune
from eval import eval_net
from unet import UNet
from utils.dataset import UNetDataset 
from utils import get_logger


def get_args():
    parser = OptionParser()
    parser.add_option('-n', '--name', dest='name',
                      default="initial", help='run name')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=2,
                      type='int', help='batch size')
    parser.add_option('-t', '--taylor_batches', dest='taylor_batches', default=500,
                      type='int', help='number of mini-batches used to calculate Taylor criterion')
    parser.add_option('-p', '--prune_channels', dest='prune_channels', default=2944, #2944
                      type='int', help='number of channels to remove')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load',
                      default="checkpoints/original/mask64_AOI_2_Vegas_Train/best.pt", help='load file model')
    parser.add_option('-r', '--lr', dest='lr', type='float',
                      default=0.0001, help='learning rate for finetuning')
    parser.add_option('-i', '--iters', dest='iters', type='int',
                      default=1000, help='number of mini-batches for fine-tuning')
    parser.add_option('-e', '--epochs', dest='epochs', type='int',
                      default=None, help='number of epochs for final finetuning')
    parser.add_option('-f', '--flops', dest='flops_reg', type='float',
                      default=.001, help='FLOPS regularization strength')
    parser.add_option('-o', '--output', dest='output', default='checkpoints/pruned',
                      help='output directory')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    # Book-keeping & paths
    args = get_args()

    # location = 'mask64_AOI_2_Vegas_Train'
    # location = 'mask64_AOI_3_Paris_Train'
    # location = 'mask64_AOI_4_Shanghai_Train'
    # location = 'mask64_AOI_5_Khartoum_Train'
    # location = 'building1'
    location = 'mask64_global_sampled_Train'


    args.load = f'checkpoints/original/{location}/best.pt'
    args.output = f'checkpoints/pruned/{location}'
    # train_im_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building2/{location}/train/images'
    # train_mask_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building2/{location}/train/masks'
    # val_im_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building2/{location}/val/images'
    # val_mask_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building2/{location}/val/masks'
    train_im_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building1/spacenet_mask_dataset64/train/images'
    train_mask_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building1/spacenet_mask_dataset64/train/masks'
    val_im_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building1/spacenet_mask_dataset64/val/images'
    val_mask_folder_path = f'/mnt/sda/nfs/rw/oec/dataset/spacenet/building1/spacenet_mask_dataset64/val/masks'
    dir_checkpoint = 'checkpoints/original/building1'
    # save_dir = os.path.join(dir_checkpoint, 'prune')
    save_dir = args.output
    os.makedirs(save_dir, exist_ok=True)
    
    log = get_logger(save_dir, 'prune')  # logger
    log.info('Args: {}'.format(json.dumps({"batch_size": args.batch_size,
                                           "taylor_batches": args.taylor_batches,
                                           "prune_channels": args.prune_channels,
                                           "gpu": args.gpu,
                                           "load": args.load,
                                           "lr": args.lr,
                                           "iters": args.iters,
                                           "epochs": args.epochs,
                                           "flops_reg": args.flops_reg},
                                          indent=4, sort_keys=True)))

    # Dataset
    train_dataset = UNetDataset(im_folder_path=train_im_folder_path, mask_folder_path=train_mask_folder_path, format='image')
    val_dataset = UNetDataset(im_folder_path=val_im_folder_path, mask_folder_path=val_mask_folder_path, format='image')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Model Initialization
    # net = UNet(in_channels=3, out_channels=1)
    net = UNet(n_channels=3, n_classes=1, f_channels='model_channels.txt')
    log.info("Built model...")
    if args.gpu:
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(args.load))
        log.info('Loading checkpoint from {}...'.format(args.load))

    # print the net's named_modules
    conv_idx = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            if name != "outconv.2":
                print(conv_idx, name, module)
                conv_idx += 1

    pruner = Pruner(net, args.flops_reg)  # Pruning handler
    criterion = nn.BCELoss()
    l2_reg_func = nn.MSELoss()

    # Ranking on the train dataset
    log.info("Evaluating Taylor criterion for %i mini-batches" % args.taylor_batches)
    with tqdm(total=args.taylor_batches*args.batch_size) as progress_bar:
        for batch_idx, (imgs, true_masks) in enumerate(train_dataloader):

            net.zero_grad()  # Zero gradients. DO NOT ACCUMULATE

            if args.gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            # Forward pass
            masks_pred = net(imgs)

            # Backward pass
            loss = criterion(masks_pred, true_masks)
            loss.backward()

            # Compute Taylor rank
            if batch_idx == 0:
                log.info("FLOPs before pruning: \n{}".format(pruner.calc_flops()))
            pruner.compute_rank()

            # Tracking progress
            progress_bar.update(args.batch_size)
            if batch_idx == args.taylor_batches:  # Stop evaluating after sufficient mini-batches
                log.info("Finished computing Taylor criterion")
                break
            break

    # Prune & save
    pruner.pruning(args.prune_channels)
    log.info('Completed Pruning of %i channels' % args.prune_channels)

    save_file = osp.join(save_dir, "pruned.pt")
    torch.save(net.state_dict(), save_file)
    log.info('Saving pruned to {}...'.format(save_file))

    save_txt = osp.join(save_dir, "pruned_channels.txt")
    pruner.channel_save(save_txt)
    log.info('Pruned channels to {}...'.format(save_txt))


    del net, pruner
    net = UNet(n_channels=3, n_classes=1, f_channels=save_txt)
    log.info("Re-Built model using {}...".format(save_txt))
    if args.gpu:
        net.cuda()
    if args.load:
        net.load_state_dict(torch.load(save_file))
        log.info('Re-Loaded checkpoint from {}...'.format(save_file))

    # optimizer = optim.SGD(net.parameters(),
    #                       lr=args.lr,
    #                       momentum=0.9,
    #                       weight_decay=0.0005)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=9.99999993923e-09, betas=(0.899999976158, 0.999000012875))


    # Use epochs or iterations for fine-tuning
    save_file = osp.join(save_dir, "finetuned.pt")

    finetune(net, optimizer, criterion, l2_reg_func, train_dataset, log, save_file,
             args.iters, args.epochs, args.batch_size, args.gpu)

    # val_dice = eval_net(net, val_dataloader, args.gpu)
    # log.info('Validation Dice Coeff: {}'.format(val_dice))


