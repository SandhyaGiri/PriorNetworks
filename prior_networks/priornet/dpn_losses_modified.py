from typing import Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F


class MixedLoss:
    """
    Computes a weighted combination of the losses provided in 'losses' param, with the weights
    of this combination taken directly from the param 'mixing_params'.

    For example: for adversarial training, Loss = 0.5 * CrossEntropyLoss(x,y,w) + 0.5 * CrossEntropyLoss(^x,y,w)
    where ^x is the adversarial sample generated from x.

    """
    def __init__(self, losses, mixing_params: Optional[Iterable[float]]):
        assert isinstance(losses, (list, tuple))
        assert isinstance(mixing_params, (list, tuple, np.ndarray))
        assert len(losses) == len(mixing_params)

        self.losses = losses
        if mixing_params is not None:
            self.mixing_params = mixing_params
        else:
            self.mixing_params = [1.] * len(self.losses)

    def __call__(self, logits_list, labels_list):
        return self.forward(logits_list, labels_list)

    def forward(self, logits_list, labels_list):
        total_loss = []
        for i, loss in enumerate(self.losses):
            weighted_loss = (loss(logits_list[i], labels_list[i])
                             * self.mixing_params[i])
            total_loss.append(weighted_loss)
        total_loss = torch.stack(total_loss, dim=0)
        # Normalize by target concentration, so that loss  magnitude is constant wrt lr and other losses
        return torch.sum(total_loss)


class PriorNetMixedLoss:
    """
    Returns a mixed or linear combination of the losses provided, with weights taken directly from the mixing_params.
    The losses provided should be of type : DirichletKLLoss.
    """
    def __init__(self, losses, mixing_params: Optional[Iterable[float]]):
        assert isinstance(losses, (list, tuple))
        assert isinstance(mixing_params, (list, tuple, np.ndarray))
        assert len(losses) == len(mixing_params)

        self.losses = losses
        if mixing_params is not None:
            self.mixing_params = mixing_params
        else:
            self.mixing_params = [1.] * len(self.losses)

    def __call__(self, logits_list, labels_list):
        return self.forward(logits_list, labels_list)

    def forward(self, logits_list, labels_list):
        total_loss = []
        target_concentration = 0.0
        for i, loss in enumerate(self.losses):
            if loss.target_concentration > target_concentration:
                target_concentration = loss.target_concentration
            weighted_loss = (loss(logits_list[i], labels_list[i])
                             * self.mixing_params[i])
            total_loss.append(weighted_loss)
        total_loss = torch.stack(total_loss, dim=0)
        # Normalize by target concentration, so that loss  magnitude is constant wrt lr and other losses
        return torch.sum(total_loss) / target_concentration


class DirichletKLLoss:
    """
    Can be applied to any DNN model which returns logits. Computes KL divergence between two dirichlet distributions. 
    
    Note that given the logits of a DNN model, exp of this yields the concentration(alpha) of the dirichlet distribution
    outputted by the DNN. From the target labels provided, a "desired/expected" target dirichlet distribution's concentration parameters
    are constructed. 
    Ex: 
        when target labels are provided, a peaky dirichlet with major concentration on target class is expected.
        when target labels are not provided, a flat dirichlet with uniform alphas (low value alpha=1) is expected.

    Loss value is then just the KL diveregence between the DNN's dirichlet and the expected dirichlet distribution.
    """

    def __init__(self, target_concentration=1e3, concentration=1.0, reverse=True):
        """
        :param target_concentration: The concentration parameter for the
        target class (if provided, needed for in domain samples)
        :param concentration: The 'base' concentration parameters for
        non-target classes. (flat dirichlet distribution with concentration(alpha)=1 by default for OOD samples, 
        or epsilon value for )
        """
        self.target_concentration = torch.tensor(target_concentration,
                                                 dtype=torch.float32)
        self.concentration = concentration
        self.reverse = reverse

    def __call__(self, logits, labels, reduction='mean'):
        logits = logits - torch.max(logits, dim=0)[0]
        alphas = torch.exp(logits)
        return self.forward(alphas, labels, reduction=reduction)

    def forward(self, alphas, labels, reduction='mean'):
        loss = self.compute_loss(alphas, labels)

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError

    def compute_loss(self, alphas, labels: Optional[torch.tensor] = None):
        """
        :param alphas: The alpha parameter outputs from the model
        :param labels: Optional. The target labels indicating the correct
        class.

        The loss creates a set of target alpha (concentration) parameters
        with all values set to self.concentration, except for the correct
        class (if provided), which is set to self.target_concentration
        :return: an array of per example loss
        """
        # TODO: Need to make sure this actually works right...
        # todo: so that concentration is either fixed, or on a per-example setup
        # Create array of target (desired) concentration parameters
        print("Given dim alphas: ", alphas.shape, " labels: ", labels.shape if labels is not None else '')
        k = alphas.shape[1] # num_classes
        if labels is None:
            # ood sample, set all alphas to 1 to get a flat simplex
            target_alphas = torch.ones_like(alphas)
        else:
            # in domain sample
            precision = 1000 # alpha_0 in paper
            epsilon = 1e-8
            target_alphas = torch.ones_like(alphas) * epsilon # consider concentration as epsilon smoothing param in paper
            target_alphas = torch.clone(target_alphas).scatter_(1, labels[:, None],
                                                               1-(k-1)*epsilon)
            # target_alphas = torch.zeros_like(alphas).scatter_(1, labels[:, None], 1)
            target_alphas *= precision
        print("target alphas:", target_alphas[0])
        if self.reverse:
            loss = dirichlet_reverse_kl_divergence(alphas=alphas, target_alphas=target_alphas)
        else:
            loss = dirichlet_kl_divergence(alphas=alphas, target_alphas=target_alphas)
        return loss


def dirichlet_kl_divergence(alphas, target_alphas, precision=None, target_precision=None,
                            epsilon=1e-8):
    """
    This function computes the Forward KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentation parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentation parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1.
        precision is the alpha_0 value representing the sum of all alpha_c's (for a normal DNN this is denominator of softmax)
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
        target precision is the alpha_0 hyperparameter, to be chosen for target dirichlet distribution.
    :param epsilon: Smoothing factor for numercal stability. Default value is 1e-8
    :return: Tensor for Batchsize X 1 of forward KL divergences between target Dirichlet and model
    """
    if not precision:
        precision = torch.sum(alphas, dim=1, keepdim=True)
    if not target_precision:
        target_precision = torch.sum(target_alphas, dim=1, keepdim=True)

    print(target_precision, precision)
    precision_term = torch.lgamma(target_precision) - torch.lgamma(precision)
    print(torch.isfinite(precision_term))
    assert torch.all(torch.isfinite(precision_term)).item()
    alphas_term = torch.sum(torch.lgamma(alphas) - torch.lgamma(target_alphas)
                            + (target_alphas - alphas) * (torch.digamma(target_alphas)
                                                          - torch.digamma(
                target_precision)), dim=1, keepdim=True)
    print("alphas:", alphas_term)
    assert torch.all(torch.isfinite(alphas_term)).item()

    cost = torch.squeeze(precision_term + alphas_term)
    return cost


def dirichlet_reverse_kl_divergence(alphas, target_alphas, precision=None, target_precision=None,
                                    epsilon=1e-8):
    """
    This function computes the Reverse KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentation parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentation parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
    :param epsilon: Smoothing factor for numercal stability. Default value is 1e-8
    :return: Tensor for Batchsize X 1 of reverse KL divergences between target Dirichlet and model
    """
    return dirichlet_kl_divergence(alphas=target_alphas, target_alphas=alphas,
                                   precision=target_precision,
                                   target_precision=precision, epsilon=epsilon)

#
# class DirichletKLLossJoint:
#     """
#     Can be applied to any model which returns logits
#
#     """
#
#     def __init__(self, concentration=1.0, reverse=True):
#         """
#         :param target_concentration: The concentration parameter for the
#         target class (if provided)
#         :param concentration: The 'base' concentration parameters for
#         non-target classes.
#         """
#
#         self.concentration = concentration
#         self.reverse = reverse
#
#     def __call__(self, logits, labels, target_concentration, gamma, mean=True):
#         alphas = torch.exp(logits)
#         return self.forward(alphas, labels, target_concentration, gamma, mean)
#
#     def forward(self, alphas, labels, target_concentration, gamma, mean):
#         loss = self.compute_loss(alphas, labels, target_concentration, gamma)
#
#         if mean:
#             return torch.mean(loss)
#         else:
#             return loss
#
#     def compute_loss(self, alphas, labels, target_concentration, gamma):
#         """
#         :param alphas: The alpha parameter outputs from the model
#         :param labels: Optional. The target labels indicating the correct
#         class.
#
#         The loss creates a set of target alpha (concentration) parameters
#         with all values set to self.concentration, except for the correct
#         class (if provided), which is set to self.target_concentration
#         :return: an array of per example loss
#         """
#         # Create array of target (desired) concentration parameters
#
#         target_concentration = target_concentration.to(alphas)
#         gamma = gamma.to(alphas)
#
#         target_alphas = torch.ones_like(alphas) * self.concentration
#         # if labels is not None:
#         target_alphas += torch.zeros_like(alphas).scatter_(1, labels[:, None],
#                                                            target_concentration[:, None])
#
#         if self.reverse:
#             loss = dirichlet_reverse_kl_divergence(alphas, target_alphas=target_alphas)
#         else:
#             loss = dirichlet_kl_divergence(alphas, target_alphas=target_alphas)
#
#         return gamma * loss
