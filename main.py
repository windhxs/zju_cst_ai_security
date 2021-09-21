import argparse
import os
import time
import torch.multiprocessing as mp

def setting(args):
    args.batch_size = getattr(args, 'batch_size', 1)
    args.epoch = getattr(args, 'epoch', 1)
    args.report_per_epoch = getattr(args, 'report_per_epoch', 50)
    # args.accum_grad = getattr(args, 'accum_grad', 10)
    # args.warmup = getattr(args, 'warmup', 100)
    
def run(gpuid, args):
    pass

def main(args):
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args, ), nprocs=len(args.gpuid), join=True)
        # The function(run) is called as fn(i, *args),
        #  where i is the process index and args is the passed through tuple of arguments.
    else :
        run(args)
if __name__ == 'main':
    parser = argparse.ArgumentParser(description='Training Parameter on SIFAR10')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpuid', nargs='+', type=int, default=0)
    parser.add_argument('--log_file', type=str, default=time.strftime("%Y_%m_%d", time.localtime()) + '.log')
    parser.add_argument("-p", "--port", type=int, default=12333)
    
    args = parser.parse_args()
    main(args)