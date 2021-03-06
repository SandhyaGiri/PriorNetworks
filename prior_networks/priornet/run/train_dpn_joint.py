import context
import argparse
import os
import sys
import pathlib
from pathlib import Path
import math
import numpy as np

import torch
from torch.utils import data
from prior_networks.priornet.dpn_losses import DirichletKLLoss, DirichletKLLossJoint
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.priornet.training import TrainerWithOODJoint
from prior_networks.util_pytorch import TargetTransform, choose_optimizer
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
parser.add_argument('--lr_decay', type=float, default=0.2, help='LR decay multiplies')
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
parser.add_argument('--FKL',
                    action='store_true',
                    help='Whether to use forward KL-divergence.')
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
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_dpn_joint.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    model_dir = Path(args.model_dir)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = model_dir / 'model'

    # Check that we are training on a sensible GPU
    assert max(args.gpu) <= torch.cuda.device_count() - 1

    device = select_gpu(args.gpu)
    # Load up the model
    ckpt = torch.load(model_dir / 'model/model.tar', map_location=device)
    model = ModelFactory.model_from_checkpoint(ckpt)
    if len(args.gpu) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        print('Using Multi-GPU training.')
    model.to(device)

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
                                                      rotation=args.rotate,
                                                      jitter=args.jitter),
                                                  target_transform=TargetTransform(args.target_concentration,
                                                                                   1.0),
                                                  download=True,
                                                  split='train')

    val_dataset = DATASET_DICT[args.id_dataset](root=args.data_path,
                                                transform=construct_transforms(
                                                    n_in=ckpt['n_in'],
                                                    mean=mean,
                                                    std=std,
                                                    mode='eval',
                                                    rotation=args.rotate,
                                                    jitter=args.jitter),
                                                target_transform=TargetTransform(args.target_concentration,
                                                                                 1.0),
                                                download=True,
                                                split='val')

    if args.gamma > 0.0:
        # Load the out-of-domain training dataset
        ood_dataset = DATASET_DICT[args.ood_dataset](root=args.data_path,
                                                     transform=construct_transforms(
                                                         n_in=ckpt['n_in'],
                                                         mean=mean,
                                                         std=std,
                                                         mode='ood'),
                                                     target_transform=TargetTransform(0.0,
                                                                                      args.gamma,
                                                                                      ood=True),
                                                     download=True,
                                                     split='train')
        ood_val_dataset = DATASET_DICT[args.ood_dataset](root=args.data_path,
                                                         transform=construct_transforms(
                                                             n_in=ckpt['n_in'],
                                                             mean=mean,
                                                             std=std,
                                                             mode='eval'),
                                                         target_transform=TargetTransform(0.0,
                                                                                          args.gamma,
                                                                                          ood=True),
                                                         download=True,
                                                         split='val')

        # Combine ID and OOD evaluation datasets into a single dataset
        assert len(val_dataset) == len(ood_val_dataset)
        print(f"Validation dataset length: {len(val_dataset)}")
        val_dataset = data.ConcatDataset([val_dataset, ood_val_dataset])

        # Even out dataset length and combine into one.
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
        print(f"Train dataset length: {len(train_dataset)}")
        train_dataset = data.ConcatDataset([train_dataset, ood_dataset])

    # Set up training and test criteria
    if args.FKL:
        criterion = DirichletKLLossJoint(concentration=args.concentration,
                                         reverse=False)
    else:
        criterion = DirichletKLLossJoint(concentration=args.concentration,
                                         reverse=True)

    # Select optimizer and optimizer params
    optimizer, optimizer_params = choose_optimizer(args.optimizer,
                                                   args.lr,
                                                   args.weight_decay)

    # Setup model trainer and train model
    lrc = [int(lrc / id_ratio) for lrc in args.lrc]
    trainer = TrainerWithOODJoint(model=model,
                                  criterion=criterion,
                                  test_criterion=criterion,
                                  train_dataset=train_dataset,
                                  test_dataset=val_dataset,
                                  optimizer=optimizer,
                                  device=device,
                                  checkpoint_path=checkpoint_path,
                                  scheduler=optim.lr_scheduler.MultiStepLR,
                                  optimizer_params=optimizer_params,
                                  scheduler_params={'milestones': lrc, 'gamma': args.lr_decay},
                                  batch_size=args.batch_size,
                                  clip_norm=args.clip_norm)
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
