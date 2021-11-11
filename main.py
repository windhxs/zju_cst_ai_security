import argparse
from operator import mod
import os
import time
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import numpy as np
import random
import torch.distributed as dist
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torchvision.transforms as tfs
from model import CIFAR10Model
from torch.optim import Adam
from log import init_logger


def setting(args):
    args.batch_size = getattr(args, 'batch_size', 256) * len(args.gpuid)
    args.epoch = getattr(args, 'epoch', 50)
    args.report_freq = getattr(args, 'report_freq', 50)
    # args.accum_grad = getattr(args, 'accum_grad', 10)
    # args.warmup = getattr(args, 'warmup', 100)
    args.seed = getattr(args, 'seed', 12345)

def run(args):
    setting(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)

    transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10('data/', train=True, transform=transform)
    logger = init_logger()
    if is_mp:
        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=False)

    else :
        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    model = CIFAR10Model(10)

    if args.cuda :
        if len(args.gpuid) == 1:
            model.to(f'cuda:{args.gpuid[0]}')
        else :
            model = nn.parallel.DataParallel(model, device_ids=args.gpuid)
            model.to(f'cuda:{args.gpuid[0]}')

    model.train()
    optimizer = Adam(model.parameters())
    loss_fct = nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        optimizer.zero_grad()
        avg_loss = 0
        step = 0 
        for i, batch in enumerate(train_dataloader):
            if args.cuda:
                batch[0] = batch[0].to(f'cuda:{args.gpuid[0]}')
                batch[1] = batch[1].to(f'cuda:{args.gpuid[0]}')
            output = model(batch[0])
            step += 1 
            loss = loss_fct(output, batch[1])

            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if ((step + 1) % args.report_freq) == 0 :
                logger.info(f'epoch: {epoch}, step: {step+1}, loss: {avg_loss}')
                avg_loss = 0
        torch.save(model.module.state_dict(), f'{args.model_path}/model_epoch_{epoch}.pt')

def test(args):
    setting(args)
    logger = init_logger()
    transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    gpuid = args.gpuid[0]
    test_set = torchvision.datasets.CIFAR10('data/', train=False, transform=transform)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    outcome = []
    for model_path in os.listdir(args.model_path):
        model = CIFAR10Model(num_classes=10)
        model.load_state_dict(torch.load(args.model_path + '/' +  model_path,  map_location=f'cuda:{gpuid}'))
        model.to(gpuid)
        model.eval()
        acc = 0
        tot = 0
        tmp = []
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                batch[0] = batch[0].to(f'cuda:{gpuid}')
                batch[1] = batch[1].to(f'cuda:{gpuid}')
                
                output = model(batch[0])
                _, pred = torch.max(output.data, 1)
                tot += batch[1].shape[0]
                acc += (pred == batch[1]).sum().item()
                tmp.append(pred) 
                
        
        outcome.append(torch.cat(tmp).tolist())
        
        logger.info(f'model from {model_path} | acc: {acc / tot * 100}%')
    
    logger.info('Ensembling..')
    outcome = torch.Tensor(outcome)

    logger.info(f'ensembled model | acc {(torch.mode(outcome, 0)[0] == torch.Tensor(test_set.targets)).sum().item() / len(test_set.targets) * 100}%' )

def main(args):
    if args.train:
        run(args)
    else:
        test(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Parameter on SIFAR10')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpuid', nargs='+', type=int, default=0)
    parser.add_argument('--log_file', type=str, default=time.strftime("%Y_%m_%d", time.localtime()) + '.log')
    parser.add_argument("-p", "--port", type=int, default=12333)
    parser.add_argument('--train', action='store_true')    
    parser.add_argument('--model_path', type=str, default='/home/hxs/zju_cst_ai_security/models/')
    args = parser.parse_args()
    main(args)