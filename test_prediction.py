import os
import torch
import time
import argparse
import shutil
import numpy as np
import torch.backends.cudnn as cudnn
from core import models
from core import datasets
from core.utils.optim import Optim
from core.utils.config import Config
from core.utils.eval import EvalPSNR
from core.ops.sync_bn_cupy.sync_bn_module import DataParallelwithSyncBN

best_PSNR = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Train Voxel Flow')
    parser.add_argument('config', help='config file path')
    args = parser.parse_args()
    return args


def main():
    checkpoint_path = '/DATA/wangshen_data/UCF101/voxelflow_finetune_model_best.pth.tar'
    global cfg, best_PSNR
    args = parse_args()
    cfg = Config.from_file(args.config)
    str1 =  ','.join(str(gpu) for gpu in cfg.device)
    print(str1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str1
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    cudnn.benchmark = True
    cudnn.fastest = True

    if hasattr(datasets, cfg.dataset):
        ds = getattr(datasets, cfg.dataset)
    else:
        raise ValueError('Unknown dataset ' + cfg.dataset)

    model = getattr(models, cfg.model.name)(cfg.model).cuda()
    cfg.train.input_mean = model.input_mean
    cfg.train.input_std = model.input_std
    cfg.test.input_mean = model.input_mean
    cfg.test.input_std = model.input_std

    # Data loading code

    val_loader = torch.utils.data.DataLoader(
        datasets.UCF101Test_NEW(cfg.test),
        batch_size=1,
        shuffle=False,
        num_workers=0, #32,
        pin_memory=True)

    if os.path.isfile(checkpoint_path):
        print(("=> loading checkpoint '{}'".format(checkpoint_path)))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'], False)
    else:
        print(("=> no checkpoint found at '{}'".format(checkpoint_path)))

    model = DataParallelwithSyncBN(
        model, device_ids=range(len(cfg.device))).cuda()

    # define loss function (criterion) optimizer and evaluator
    criterion = torch.nn.MSELoss().cuda()
    evaluator = EvalPSNR(255.0 / np.mean(cfg.test.input_std))


    PSNR = validate(val_loader, model, criterion, evaluator)

    print(PSNR)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1),
               -1)[:,
                   getattr(
                       torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[
                           x.is_cuda])().long(), :]
    return x.view(xsize)


def validate(val_loader, model, criterion, evaluator):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        evaluator.clear()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)

            loss = criterion(output, target_var)

            # measure accuracy and record loss

            pred = output.data.cpu().numpy()
            evaluator(pred, target.cpu().numpy())
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            print(('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'PSNR {PSNR:.3f}'.format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        PSNR=evaluator.PSNR())))

        print('Testing Results: '
              'PSNR {PSNR:.3f} ({bestPSNR:.4f})\tLoss {loss.avg:.5f}'.format(
                  PSNR=evaluator.PSNR(),
                  bestPSNR=max(evaluator.PSNR(), best_PSNR),
                  loss=losses))

        return evaluator.PSNR()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not cfg.output_dir:
        return
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    filename = os.path.join(cfg.output_dir, '_'.join((cfg.snapshot_pref,
                                                      filename)))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(cfg.output_dir, '_'.join(
            (cfg.snapshot_pref, 'model_best.pth.tar')))
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
