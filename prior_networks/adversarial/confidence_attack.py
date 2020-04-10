#! /usr/bin/env python
import context
import argparse
import os
import sys
import numpy as np

from PIL import Image

import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path

from prior_networks.assessment.misc_detection import eval_misc_detect
from prior_networks.evaluation import eval_logits_on_dataset
from prior_networks.datasets.image import construct_transforms
from prior_networks.datasets.image.dataspliter import DataSpliter
from prior_networks.assessment.calibration import classification_calibration
from prior_networks.assessment.rejection import eval_rejection_ratio_class
from prior_networks.priornet.dpn import dirichlet_prior_network_uncertainty
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.models.model_factory import ModelFactory
from prior_networks.adversarial.fgm import construct_fgm_attack
from prior_networks.adversarial.pgd import construct_pgd_attack

matplotlib.use('agg')

parser = argparse.ArgumentParser(description='Constructs an FGSM/PGD attack on test images and \
                    reports the results as epsilon vs adversarial success rate.')
parser.add_argument('data_path', type=str,
                    help='Path where data is saved')
parser.add_argument('dataset', choices=DATASET_DICT.keys(),
                    help='Specify name of dataset to evaluate model on.')
parser.add_argument('output_path', type=str,
                    help='Path of directory for saving model outputs.')
parser.add_argument('--epsilon','--list', nargs='+', type=float,
                    help='Strength perturbation in range of 0 to 1, ex: 0.25', required=True)
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for processing')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specify which GPUs to to run on.')
parser.add_argument('--n_channels', type=int, default=3,
                    help='Choose number in image channels. Default 3 for color images.')
parser.add_argument('--train', action='store_true',
                    help='Whether to evaluate on the training data instead of test data')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite a previous run of this script')
parser.add_argument('--attack_type', type=str, choices=['FGSM', 'PGD'], default='FGSM',
                    help='Choose the type of attack to be performed.')
parser.add_argument('--norm', type=str, choices=['inf', '2'], default='inf',
                    help='The type of norm ball to restrict the adversarial samples to. Needed only for PGD attack.')
parser.add_argument('--step_size', type=float, default=0.4,
                    help='The size of the gradient update step, used during each iteration in PGD attack.')
parser.add_argument('--max_steps', type=int, default=10,
                    help='The number of the gradient update steps, to be performed for PGD attack.')

from typing import Optional, Tuple

def confidence_criteria(outputs, labels):
    """
        Calculates the confidence loss and returns it.

        Note:
        Ideally we want to maximize confidence_of_model (optimization objective)
        But as a loss function we should minimize  -1 * confidence_of_model, to achieve the same objective as above.
    """
    outputs = outputs - torch.max(outputs, dim=0)[0] # numerically stable softmax 
    alphas = torch.exp(outputs)
    alpha0 = torch.sum(alphas, dim=1, keepdim=True)
    probs = torch.div(alphas, alpha0)
    return torch.neg(torch.max(probs, dim=1, keepdim=True)[0])

def construct_adversarial_dataset(model: nn.Module, epsilon, dataset: Dataset,
                                  attack_type, norm, step_size, max_steps, batch_size: int = 128,
                                  device: Optional[torch.device] = None,
                                  num_workers: int = 4):
    """
    Takes a model and an evaluation dataset, and returns the logits
    output by the model on that dataset (adversarial samples generated from it) as an array
    :param model: torch.nn.Module that outputs model logits
    :param dataset: pytorch dataset with inputs and labels
    :param batch_size: int
    :param device: device to use for evaluation
    :param num_workers: int, num. workers for the data loader
    :return: stacked torch tensor of logits returned by the model
    on the adversarial dataset, the corresponding labels and the generated adversarial images.
    """
    # Set model in eval mode
    model.eval()

    testloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    logits_list = []
    labels_list = []
    adv_list = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, labels = data
            if device is not None:
                inputs, labels = map(lambda x: x.to(device),
                                     (inputs, labels))
            if attack_type == 'FGSM':
                adv_inputs = construct_fgm_attack(model=model, labels=labels, inputs=inputs, epsilon=epsilon, 
                                criterion=confidence_criteria, device=device)
            elif attack_type == 'PGD':
                adv_inputs = construct_pgd_attack(model=model, labels=labels, inputs=inputs, epsilon=epsilon, 
                                criterion=confidence_criteria, device=device,
                                norm=norm, step_size=step_size, max_steps=max_steps)
            model.zero_grad()
            logits = model(adv_inputs)

            logits_list.append(logits)
            labels_list.append(labels)
            adv_list.append(adv_inputs.detach())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    adv = torch.cat(adv_list, dim=0)

    return logits.cpu(), labels.cpu(), adv.cpu()

def perform_epsilon_attack(model, epsilon, dataset, correct_classified_indices, 
                            batch_size, device, n_channels, output_path, mean, std, attack_type, **kwargs):
    assert 0.0 < epsilon <= 1.0
    
    logits, labels, images = construct_adversarial_dataset(model=model,
                                                           dataset=dataset,
                                                           epsilon=epsilon,
                                                           batch_size=batch_size,
                                                           device=device,
                                                           attack_type=attack_type,
                                                           step_size=kwargs.get('step_size'),
                                                           norm=kwargs.get('norm'),
                                                           max_steps=kwargs.get('max_steps'))
    labels, probs, logits = labels.numpy(), F.softmax(logits, dim=1).numpy(), logits.numpy()
    images = images.numpy()
    print(f"Epsilon {epsilon} {attack_type} attack. Adv images: {images.shape}")

    # Save model outputs
    np.savetxt(os.path.join(output_path, 'labels.txt'), labels)
    np.savetxt(os.path.join(output_path, 'probs.txt'), probs)
    np.savetxt(os.path.join(output_path, 'logits.txt'), logits)

    # determine misclassifications under attack
    preds = np.argmax(probs, axis=1)
    misclassifications = np.asarray(preds != labels, dtype=np.int32)
    misclassified_indices = np.argwhere(misclassifications == 1)

    # count as success only those samples that are wrongly classified under attack, while they were correctly classified without attack.
    adv_success = len(np.intersect1d(correct_classified_indices, misclassified_indices))

    # save perturbed or adversarial images.
    persist_images(images, mean, std, n_channels, os.path.join(output_path, "adv-images"))

    # Get dictionary of uncertainties.
    uncertainties = dirichlet_prior_network_uncertainty(logits)

    # Save uncertainties
    for key in uncertainties.keys():
        np.savetxt(os.path.join(output_path, key + '.txt'), uncertainties[key])

    nll = -np.mean(np.log(probs[np.arange(probs.shape[0]), np.squeeze(labels)] + 1e-10))

    accuracy = np.mean(np.asarray(labels == np.argmax(probs, axis=1), dtype=np.float32))
    with open(os.path.join(output_path, 'results.txt'), 'a') as f:
        f.write(f'Classification Error: {np.round(100 * (1.0 - accuracy), 1)} \n')
        f.write(f'NLL: {np.round(nll, 3)} \n')

    # Assess Misclassification Detection
    eval_misc_detect(labels, probs, uncertainties, save_path=output_path, misc_positive=True)

    # return aversarial successes
    return adv_success

def persist_images(images, mean, std, n_channels, image_dir):
    assert type(images) == np.ndarray, ("Unexpected input type. Cannot be persisted.")
    # Images to be in [-1, 1] interval, so rescale them back to [0, 1].
    images= np.asarray((images*std + mean)*255.0, dtype=np.uint8)

    for i, image in enumerate(images):
        if n_channels == 1:
            # images were added new channels (3) to go through VGG, so remove unnecessary channels
            Image.fromarray(image[0,:,:]).save(os.path.join(image_dir, f"{i}.png"))
        else:
            Image.fromarray(image).save(os.path.join(image_dir, f"{i}.png"))

def persist_dataset(dataset, mean, std, n_channels, image_dir):
    assert isinstance(dataset, torch.utils.data.Dataset), ("Dataset is not of right type. Cannot be persisted.")
    images = []
    testloader = DataLoader(dataset, batch_size=128,
                            shuffle=False, num_workers=4)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            image, _ = data
            images.append(image)
    
    persist_images(torch.cat(images, dim=0).numpy(), mean, std, n_channels, image_dir)

def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/construct_adversarial_dataset.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    if os.path.isdir(args.output_path) and not args.overwrite:
        print(f'Directory {args.output_path} exists. Exiting...')
        sys.exit()
    elif os.path.isdir(args.output_path) and args.overwrite:
        os.remove(args.output_path + '/*')
    else:
        os.makedirs(args.output_path)

    # Check that we are using a sensible GPU
    device = select_gpu(args.gpu)

    # Load up the model
    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / 'model/model.tar', map_location=device)
    model = ModelFactory.model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()

    # Load the in-domain evaluation data
    if args.train:
        dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                             transform=construct_transforms(n_in=ckpt['n_in'],
                                                                            mean=DATASET_DICT[args.dataset].mean,
                                                                            std=DATASET_DICT[args.dataset].std,
                                                                            num_channels=args.n_channels,
                                                                            mode='train'),
                                             target_transform=None,
                                             download=True,
                                             split='train')
    else:
        dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                             transform=construct_transforms(n_in=ckpt['n_in'],
                                                                            mean=DATASET_DICT[args.dataset].mean,
                                                                            std=DATASET_DICT[args.dataset].std,
                                                                            num_channels=args.n_channels,
                                                                            mode='eval'),
                                             target_transform=None,
                                             download=True,
                                             split='test')
    
    # dataset = DataSpliter.reduceSize(dataset, 16)

    mean = np.array(DATASET_DICT[args.dataset].mean).reshape((3, 1, 1))
    std = np.array(DATASET_DICT[args.dataset].std).reshape((3, 1, 1))

    # dataset = torch.utils.data.Subset(dataset, image_indices)
    print("dataset length:", len(dataset))

    org_dataset_folder = os.path.join(args.output_path, "org-images")
    os.makedirs(org_dataset_folder)
    persist_dataset(dataset, mean, std, args.n_channels, org_dataset_folder)

    # perform original evaluation on the model using unperturbed images
    logits, labels = eval_logits_on_dataset(model, dataset=dataset, device=device, batch_size=args.batch_size)
    labels = labels.numpy()
    # determine correct classifications without attack (original non perturbed images)
    org_preds = np.argmax(F.softmax(logits, dim=1).numpy(), axis=1)
    correct_classifications = np.asarray(org_preds == labels, dtype=np.int32)
    correct_classified_indices = np.argwhere(correct_classifications == 1)

    # perform attacks on the same dataset, using different epsilon values.
    adv_success_rates = []
    for epsilon in args.epsilon:
        attack_folder = os.path.join(args.output_path, f"e{epsilon}-attack")
        out_path = os.path.join(attack_folder, "adv-images")
        os.makedirs(out_path)
        adv_success = perform_epsilon_attack(model, epsilon, dataset, correct_classified_indices, args.batch_size, 
                            device, args.n_channels,attack_folder, mean, std, args.attack_type,
                            norm=args.norm, step_size=args.step_size, max_steps=args.max_steps)
        adv_success_rates.append(adv_success / len(correct_classified_indices))
    
    # plot the epsilon, adversarial success rate graph (line plot)
    plt.figure(figsize=(5,5))
    plt.plot(args.epsilon, adv_success_rates, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(np.min(args.epsilon), np.max(args.epsilon), step=0.1))
    plt.title("Adversarial Success Rate vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Adversarial Success Rate")
    plt.savefig(os.path.join(args.output_path, "epsilon-curve.png"))

if __name__ == '__main__':
    main()
