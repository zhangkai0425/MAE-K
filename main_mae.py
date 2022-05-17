import time
import math
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from functools import partial
import tensorboard_logger
from deit import DistilledVisionTransformer
from vit import ViT
from model import MAE
from util import *

# for re-produce
set_seed(0)


def build_model(args):
    '''
    build MAE model.
    :param args: model args
    :return: model
    '''
    # build model
    v = ViT(image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.n_class,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim)

    DeiT_Tiny = DistilledVisionTransformer(
            img_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.n_class,
            embed_dim=48, #之后改192
            depth=args.vit_depth,
            num_heads=3,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )

    mae = MAE(encoder=DeiT_Tiny,
              masking_ratio=args.masking_ratio,
              decoder_dim=args.decoder_dim,
              decoder_depth=args.decoder_depth,
              device=args.device).to(args.device)

    return mae


def train(args):
    '''
    train the model
    :param args: parameters
    :return:
    '''
    # load data
    data_loader, args.n_class = load_data(args.data_dir,
                                          args.data_name,
                                          image_size=args.image_size,
                                          batch_size=args.batch_size,
                                          n_worker=args.n_worker,
                                          is_train=True)
                
    print("data already prepared!")
    # build mae model
    model = build_model(args)
    model.train()
    print("model already prepared!")
    # build optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.base_lr,
                                  weight_decay=args.weight_decay,
                                  betas=args.momentum)
    print("optimizer already prepared!")
    # learning rate scheduler: warmup + consine
    def lr_lambda(epoch):
        if epoch < args.epochs_warmup:
            p = epoch / args.epochs_warmup
            lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
        else:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        return lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # tensorboard
    tb_logger = tensorboard_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    for epoch in range(1, args.epochs + 1):
        # records
        ts = time.time()
        losses = AverageMeter()
        print("training at epoch:",epoch)
        # train by epoch
        for idx, (images, targets) in enumerate(tqdm(data_loader)):
            # put images into device
            images = images.to(args.device)
            # forward
            loss = model(images)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # record
            losses.update(loss.item(), args.batch_size)
            # print("training at:",idx," loss = ",loss)

        # log
        tb_logger.log_value('loss', losses.avg, epoch)

        # print
        if epoch % args.print_freq == 0:
            print('- epoch {:3d}, time, {:.2f}s, loss {:.4f}'.format(epoch, time.time() - ts, losses.avg))

        # save checkpoint
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.ckpt_folder, 'epoch_{:d}.ckpt'.format(epoch))
            save_ckpt(model, optimizer, args, epoch, save_file=save_file)

    # save the last checkpoint
    save_file = os.path.join(args.ckpt_folder, 'MAE.ckpt')
    save_ckpt(model, optimizer, args, epoch, save_file=save_file)

def train_k(args):
    '''
    train the model MAE-K
    :param args: parameters
    :return:
    '''
    # load data
    data_loader, args.n_class = load_data(args.data_dir,
                                          args.data_name,
                                          image_size=args.image_size,
                                          batch_size=args.batch_size,
                                          n_worker=args.n_worker,
                                          is_train=True)
    print("data already prepared!")
    # print("class:",args.n_class)
    # build mae model
    model = build_model(args)
    model.train()
    print("model already prepared!")
    # build optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.base_lr,
                                  weight_decay=args.weight_decay,
                                  betas=args.momentum)
    print("optimizer already prepared!")
    # learning rate scheduler: warmup + consine
    def lr_lambda(epoch):
        if epoch < args.epochs_warmup:
            p = epoch / args.epochs_warmup
            lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)
        else:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            lr = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        return lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # tensorboard
    tb_logger = tensorboard_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    K = args.K 
    epoch_turn = args.epochs//K

    # MAE-1 training 
    print("training MAE-1")
    for epoch in range(1, epoch_turn + 1):
        ts = time.time()
        losses = AverageMeter()
        print("training at epoch:",epoch)
        # train by epoch
        for idx, (images, targets) in enumerate(tqdm(data_loader)):
            # put images into device
            images = images.to(args.device)
            # forward
            loss = model(images)
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # record
            losses.update(loss.item(), args.batch_size)
        # log
        tb_logger.log_value('loss', losses.avg, epoch)
        # print
        if epoch % args.print_freq == 0:
            print('- epoch {:3d}, time, {:.2f}s, loss {:.4f}'.format(epoch, time.time() - ts, losses.avg))
        # save checkpoint
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.ckpt_folder, 'epoch_{:d}.ckpt'.format(epoch))
            save_ckpt(model, optimizer, args, epoch, save_file=save_file)
    # save the last checkpoint
    save_file = os.path.join(args.ckpt_folder, 'MAE-1.ckpt')
    save_ckpt(model, optimizer, args, epoch, save_file=save_file)

    # save MAE-1 for the next step:the most trivial method
    torch.save(model,'ckpt/tmp/model_pre.pkl')
    print("MAE-1 model saved!")
    # train for 1 + (k-1) times
    for ki in range(2,K+1):
        print("training MAE-{:d}".format(ki))
        model_pre = torch.load('ckpt/tmp/model_pre.pkl')
        print("pre-model already loaded!")
        for epoch in range(1, epoch_turn + 1):
            # records
            ts = time.time()
            losses = AverageMeter()
            print("training at epoch:",epoch+epoch_turn*(ki-1))
            # train by epoch
            for idx, (images, targets) in enumerate(tqdm(data_loader)):
                # put images into device
                images = images.to(args.device)
                # forward
                loss = model.forward_k(images,model_pre)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                # record
                losses.update(loss.item(), args.batch_size)
                # print("training at:",idx," loss = ",loss)
            # log
            tb_logger.log_value('loss', losses.avg, epoch+epoch_turn*(ki-1))
            # print
            if (epoch+epoch_turn*(ki-1)) % args.print_freq == 0:
                print('- epoch {:3d}, time, {:.2f}s, loss {:.4f}'.format(epoch+epoch_turn*(ki-1), time.time() - ts, losses.avg))
            # save checkpoint
            if (epoch+epoch_turn*(ki-1)) % args.save_freq == 0:
                save_file = os.path.join(args.ckpt_folder, 'epoch_{:d}.ckpt'.format(epoch+epoch_turn*(ki-1)))
                save_ckpt(model, optimizer, args, epoch, save_file=save_file)
        # save the last checkpoint
        save_file = os.path.join(args.ckpt_folder, 'MAE-%s.ckpt'%ki)
        save_ckpt(model, optimizer, args, epoch, save_file=save_file)
        torch.save(model,'ckpt/tmp/model_pre.pkl')
    print("Training MAE-{:d} Completed".format(args.K))

def default_args(data_name, trail='MAE'):
    '''
    for default parameters. tune them upon your options
    :param data_name: dataset name, such as 'imagenet'
    :param trail: an int indicator to specify different runnings
    :return:
    '''
    # params
    args = argparse.ArgumentParser().parse_args()

    # device
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # data
    args.data_dir = 'data'
    args.data_name = data_name
    args.image_size = 32
    args.n_worker = 8

    # model
    # - use ViT-Base whose parameters are referred from "Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers
    # - for Image Recognition at Scale. ICLR 2021. https://openreview.net/forum?id=YicbFdNTTy".
    args.patch_size = 4 # for cifar-10,we use patch_size as 4
    args.vit_dim = 48
    args.vit_depth = 12
    args.vit_heads = 12
    args.vit_mlp_dim = 3072
    args.masking_ratio = 0.75  # the paper recommended 75% masked patches
    args.decoder_dim = 512  # paper showed good results with 512
    args.decoder_depth = 8  # paper showed good results with 8

    # train
    args.K = 200 # bootstrapped num
    args.batch_size = 512
    args.epochs = 200
    args.base_lr = 1.5e-4
    args.lr = args.base_lr * args.batch_size / 256
    args.weight_decay = 5e-2
    args.momentum = (0.9, 0.95)
    args.epochs_warmup = 40
    args.warmup_from = 1e-4
    args.lr_decay_rate = 1e-2
    eta_min = args.lr * (args.lr_decay_rate ** 3)
    args.warmup_to = eta_min + (args.lr - eta_min) * (1 + math.cos(math.pi * args.epochs_warmup / args.epochs)) / 2

    # print and save
    args.print_freq = 10
    args.save_freq = 10

    # tensorboard
    args.tb_folder = os.path.join('log', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)

    # ckpt
    args.ckpt_folder = os.path.join('ckpt', '{}_{}'.format(args.data_name, trail))
    if not os.path.isdir(args.ckpt_folder):
        os.makedirs(args.ckpt_folder)

    return args


if __name__ == '__main__':

    data_name = 'cifar10'
    trail = 'MAE-200-new'
    # mode 'MAE-K' or 'MAE'
    mode = 'MAE-K'
    if mode == 'MAE':
        train(default_args(data_name,trail=trail))
    if mode == 'MAE-K':
        train_k(default_args(data_name,trail=trail))