import context
import argparse
import os
import sys
import pathlib
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from prior_networks.priornet.dpn_losses import DirichletKLLoss, PriorNetMixedLoss
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.priornet.training import TrainerWithOOD
from prior_networks.util_pytorch import TargetTransform, choose_optimizer
from prior_networks.plot_util import visualize_data
from torch import optim
from prior_networks.datasets.image.standardised_datasets import construct_transforms
from prior_networks.models.model_factory import ModelFactory

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('data_path', type=str,
                    help='absolute path to training data.')
parser.add_argument('id_dataset', choices=DATASET_DICT.keys(),
                    help='In-domain dataset name.')
parser.add_argument('ood_dataset', choices=DATASET_DICT.keys(),
                    help='Out-of-domain dataset name.')
parser.add_argument('n_epochs', type=int,
                    help='How many epochs to train for.')
parser.add_argument('lr', type=float,
                    help='Initial learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.95, help='LR decay multiplies')
parser.add_argument('--lrc', action='append', type=int, help='LR decay milestones')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--target_concentration', type=float, default=1e2,
                    help='Target in-domain concentration.')
parser.add_argument('--concentration', type=float, default=1.0,
                    help='Concentration of non-target classes.')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='Weight for OOD loss.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='L2 weight decay.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')
parser.add_argument('--model_load_path', type=str, default='./model',
                    help='Source where to load the model from.')
parser.add_argument('--reverse_KL',
                    action='store_true',
                    help='Whether to use forward or reverse KL. Default is to ALWAYS use forward KL.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specify which GPUs to to run on.')
parser.add_argument('--optimizer', choices=['SGD', 'ADAM'], default='SGD',
                    help='Choose which optimizer to use.')
parser.add_argument('--augment',
                    action='store_true',
                    help='Whether to use augmentation.')
parser.add_argument('--rotate',
                    action='store_true',
                    help='Whether to use rotation augmentation')
parser.add_argument('--jitter', type=float, default=0.0,
                    help='Specify how much random color, '
                         'hue, saturation and contrast jitter to apply')
parser.add_argument('--n_channels', type=int, default=3,
                    help='Choose number in image channels. Default 3 for color images.')
parser.add_argument('--normalize',
                    action='store_false',
                    help='Whether to standardize input (x-mu)/std')
parser.add_argument('--resume',
                    action='store_true',
                    help='Whether to resume training from checkpoint.')
parser.add_argument('--clip_norm', type=float, default=10.0,
                    help='Gradient clipping norm value.')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to where to checkpoint.')


def main():
    args = parser.parse_args()
    print(args.__dict__)
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_dpn.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    model_dir = Path(args.model_dir)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = model_dir / 'model'
    # Check that we are training on a sensible GPU
    if os.environ.get('SLURM_JOB_GPUS', None) is not None:
        args.gpu = list(map(int, os.environ['SLURM_JOB_GPUS'].split(",")))
    
    if torch.cuda.is_available():
        assert len(args.gpu) <= torch.cuda.device_count()

    # Load up the model
    ckpt = torch.load(model_dir / 'model/model.tar')
    model = ModelFactory.model_from_checkpoint(ckpt)
    if len(args.gpu) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        print('Using Multi-GPU training.')
    else:
        device = select_gpu(args.gpu)
        model.to(device)

    print("Args normalize: ", args.normalize) # Needed to ensure that mean, std have right dim / channels as per dataset
    if args.normalize:
        mean = DATASET_DICT[args.id_dataset].mean
        std = DATASET_DICT[args.id_dataset].std
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    # Load the in-domain training and validation data
    train_dataset = DATASET_DICT[args.id_dataset](root=args.data_path,
                                                  transform=construct_transforms(
                                                      n_in=ckpt['n_in'],
                                                      mode='train',
                                                      mean=mean,
                                                      std=std,
                                                      augment=args.augment,
                                                      num_channels=args.n_channels,
                                                      rotation=args.rotate,
                                                      jitter=args.jitter),
                                                  target_transform=None,
                                                  download=True,
                                                  split='train')
    # take 1 training sample
    # train_dataset = torch.utils.data.random_split(train_dataset, [100, len(train_dataset)-100])[0]

    val_dataset = DATASET_DICT[args.id_dataset](root=args.data_path,
                                                transform=construct_transforms(
                                                    n_in=ckpt['n_in'],
                                                    mean=mean,
                                                    std=std,
                                                    mode='eval',
                                                    rotation=args.rotate,
                                                    num_channels=args.n_channels,
                                                    jitter=args.jitter),
                                                target_transform=None,
                                                download=True,
                                                split='val')
    # val_dataset = torch.utils.data.random_split(val_dataset, [100, len(val_dataset)-100])[0]

    # Load the out-of-domain training dataset
    ood_dataset = DATASET_DICT[args.ood_dataset](root=args.data_path,
                                                 transform=construct_transforms(
                                                     n_in=ckpt['n_in'],
                                                     mean=mean,
                                                     std=std,
                                                     num_channels=args.n_channels,
                                                     mode='ood'),
                                                 target_transform=None,
                                                 download=True,
                                                 split='train')
    # ood_dataset = torch.utils.data.random_split(ood_dataset, [100, len(ood_dataset)-100])[0]
    ood_val_dataset = DATASET_DICT[args.ood_dataset](root=args.data_path,
                                                     transform=construct_transforms(
                                                         n_in=ckpt['n_in'],
                                                         mean=mean,
                                                         std=std,
                                                         num_channels=args.n_channels,
                                                         mode='eval'),
                                                     target_transform=None,
                                                     download=True,
                                                     split='val')
    # ood_val_dataset = torch.utils.data.random_split(ood_val_dataset, [100, len(ood_val_dataset)-100])[0]

    # Combine ID and OOD training datasets into a single dataset for
    # training (necessary for DataParallel training)
    print(len(val_dataset), len(ood_val_dataset))
    if len(ood_val_dataset) > len(val_dataset):
        ood_val_dataset= torch.utils.data.random_split(ood_val_dataset, [len(val_dataset), len(ood_val_dataset)-len(val_dataset)])[0]

    print(len(val_dataset), len(ood_val_dataset))
    assert len(val_dataset) == len(ood_val_dataset)

    # Even out dataset lengths.
    id_ratio = 1.0
    if len(train_dataset) < len(ood_dataset):
        id_ratio = np.ceil(float(len(ood_dataset)) / float(len(train_dataset)))
        assert id_ratio.is_integer()
        dataset_list = [train_dataset, ] * (int(id_ratio))
        train_dataset = data.ConcatDataset(dataset_list)

    if len(train_dataset) > len(ood_dataset):
        ratio = np.ceil(float(len(train_dataset)) / float(len(ood_dataset)))
        assert ratio.is_integer()
        dataset_list = [ood_dataset, ] * int(ratio)
        ood_dataset = data.ConcatDataset(dataset_list)

        if len(ood_dataset) > len(train_dataset):
            ood_dataset = data.Subset(ood_dataset, np.arange(0, len(train_dataset)))

    assert len(train_dataset) == len(ood_dataset)
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Train dataset length: {len(train_dataset)}")

    # visualize some data
    # visualize_data(ood_dataset, filename='ood')

    # Set up training and test criteria
    id_criterion = DirichletKLLoss(target_concentration=args.target_concentration,
                                   concentration=args.concentration,
                                   reverse=args.reverse_KL)

    ood_criterion = DirichletKLLoss(target_concentration=0.0,
                                    concentration=args.concentration,
                                    reverse=args.reverse_KL)

    criterion = PriorNetMixedLoss([id_criterion, ood_criterion], mixing_params=[1.0, args.gamma])

    # Select optimizer and optimizer params
    optimizer, optimizer_params = choose_optimizer(args.optimizer,
                                                   args.lr,
                                                   args.weight_decay)

    # Setup model trainer and train model
    lrc = [int(lrc / id_ratio) for lrc in args.lrc]
    trainer = TrainerWithOOD(model=model,
                             criterion=criterion,
                             id_criterion=id_criterion,
                             ood_criterion=ood_criterion,
                             test_criterion=criterion,
                             ood_dataset=ood_dataset,
                             test_ood_dataset=ood_val_dataset,
                             train_dataset=train_dataset,
                             test_dataset=val_dataset,
                             optimizer=optimizer,
                             device=device,
                             checkpoint_path=checkpoint_path,
                             scheduler=optim.lr_scheduler.ExponentialLR,
                             optimizer_params=optimizer_params,
                             scheduler_params={'gamma': args.lr_decay},
                             batch_size=args.batch_size,
                             clip_norm=args.clip_norm,
                             log_dir=model_dir)
    if args.resume:
        try:
            trainer.load_checkpoint(True, True, map_location=device)
        except:
            print('No checkpoint found, training from empty model.')
            pass
    trainer.train(int(args.n_epochs / id_ratio), resume=args.resume)

    # Save final model
    if len(args.gpu) > 1 and torch.cuda.device_count() > 1:
        model = model.module
    ModelFactory.checkpoint_model(path=model_dir / 'model/model.tar',
                                  model=model,
                                  arch=ckpt['arch'],
                                  dropout_rate=ckpt['dropout_rate'],
                                  n_channels=ckpt['n_channels'],
                                  num_classes=ckpt['num_classes'],
                                  small_inputs=ckpt['small_inputs'],
                                  n_in=ckpt['n_in'])


if __name__ == "__main__":
    main()
