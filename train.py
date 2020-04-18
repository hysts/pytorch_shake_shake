#!/usr/bin/env python

import argparse
from collections import OrderedDict
import importlib
import json
import logging
import pathlib
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torchvision
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--base_channels', type=int, required=True)

    parser.add_argument('--shake_forward', type=str2bool, default=True)
    parser.add_argument('--shake_backward', type=str2bool, default=True)
    parser.add_argument('--shake_image', type=str2bool, default=True)

    # run config
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--device', type=str, default='cuda')

    # optim config
    parser.add_argument('--epochs', type=int, default=1800)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--lr_min', type=float, default=0)

    # TensorBoard
    parser.add_argument('--tensorboard',
                        dest='tensorboard',
                        action='store_true')

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    model_config = OrderedDict([
        ('arch', 'shake_shake'),
        ('depth', args.depth),
        ('base_channels', args.base_channels),
        ('shake_forward', args.shake_forward),
        ('shake_backward', args.shake_backward),
        ('shake_image', args.shake_image),
        ('input_shape', (1, 3, 32, 32)),
        ('n_classes', 10),
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('nesterov', args.nesterov),
        ('lr_min', args.lr_min),
    ])

    data_config = OrderedDict([
        ('dataset', 'CIFAR10'),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('outdir', args.outdir),
        ('num_workers', args.num_workers),
        ('device', args.device),
        ('tensorboard', args.tensorboard),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config


def load_model(config):
    module = importlib.import_module(config['arch'])
    Network = getattr(module, 'Network')
    return Network(config)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def _cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def get_cosine_annealing_scheduler(optimizer, optim_config):
    total_steps = optim_config['epochs'] * optim_config['steps_per_epoch']

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            optim_config['lr_min'] / optim_config['base_lr']))

    return scheduler


def train(epoch, model, optimizer, scheduler, criterion, train_loader,
          run_config, writer):
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if run_config['tensorboard'] and step == 0:
            image = torchvision.utils.make_grid(data,
                                                normalize=True,
                                                scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        if run_config['tensorboard']:
            writer.add_scalar('Train/LearningRate',
                              scheduler.get_lr()[0], global_step)

        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.item()
        correct_ = preds.eq(targets).sum().item()
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            accuracy_meter.val,
                            accuracy_meter.avg,
                        ))
        scheduler.step()

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)

    train_log = OrderedDict({
        'epoch':
        epoch,
        'train':
        OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy_meter.avg,
            'time': elapsed,
        }),
    })
    return train_log


def test(epoch, model, criterion, test_loader, run_config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()
    device = torch.device(run_config['device'])

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for step, (data, targets) in enumerate(test_loader):
            if run_config['tensorboard'] and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(data,
                                                    normalize=True,
                                                    scale_each=True)
                writer.add_image('Test/Image', image, epoch)

            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    test_log = OrderedDict({
        'epoch':
        epoch,
        'test':
        OrderedDict({
            'loss': loss_meter.avg,
            'accuracy': accuracy,
            'time': elapsed,
        }),
    })
    return test_log


def main():
    # parse command line arguments
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = pathlib.Path(run_config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    # TensorBoard SummaryWriter
    writer = SummaryWriter(
        outdir.as_posix()) if run_config['tensorboard'] else None

    # save config as json file in output directory
    outpath = outdir / 'config.json'
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # data loaders
    train_loader, test_loader = get_loader(optim_config['batch_size'],
                                           run_config['num_workers'],
                                           run_config['device'] != 'cpu')

    # model
    model = load_model(config['model_config'])
    model.to(torch.device(run_config['device']))
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    criterion = nn.CrossEntropyLoss(reduction='mean')

    optim_config['steps_per_epoch'] = len(train_loader)
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=optim_config['base_lr'],
                                momentum=optim_config['momentum'],
                                weight_decay=optim_config['weight_decay'],
                                nesterov=optim_config['nesterov'])
    scheduler = get_cosine_annealing_scheduler(optimizer, optim_config)

    # run test before start training
    test(0, model, criterion, test_loader, run_config, writer)

    epoch_logs = []
    for epoch in range(1, optim_config['epochs'] + 1):
        train_log = train(epoch, model, optimizer, scheduler, criterion,
                          train_loader, run_config, writer)
        test_log = test(epoch, model, criterion, test_loader, run_config,
                        writer)

        epoch_log = train_log.copy()
        epoch_log.update(test_log)
        epoch_logs.append(epoch_log)
        with open(outdir / 'log.json', 'w') as fout:
            json.dump(epoch_logs, fout, indent=2)

        state = OrderedDict([
            ('config', config),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('accuracy', test_log['test']['accuracy']),
        ])
        model_path = outdir / 'model_state.pth'
        torch.save(state, model_path)


if __name__ == '__main__':
    main()
