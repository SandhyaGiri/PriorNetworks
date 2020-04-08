import torch
from torch import nn


def construct_pgd_attack(model,
                         inputs,
                         labels,
                         epsilon,
                         criterion=nn.CrossEntropyLoss(),
                         device=None,
                         norm="inf",
                         step_size=0.4,
                         max_steps=10,
                         pin_memory: bool = True):
    """
        Constructs adversarial images by doing multiple iterations of criterion maximization (maximize loss) to get
        the best adversarial image within a p-norm ball around the input image.

        params
        -------
        norm - string
            can be any one of the 'p' norm values given to torch.norm function. Used for projecting the perturbed image on to the p-norm ball.
            Possible values: "inf" , "2"
        step_size - float
            indicates the size of the gradient/gradient sign update to be done at each step.
        max_steps - int
            indicates the maximum steps to perform for chosing the best adversary (one with max loss/criterion).
    """
    adv_inputs = inputs.clone()
    print(adv_inputs.is_leaf)
    adv_inputs.requires_grad = True
    model.eval()

    max_loss = None
    best_adversary = None

    epsilon = torch.ones([inputs.size()[0]]) * epsilon
    epsilon = epsilon.view([epsilon.size()[0], 1, 1, 1]) # transform to a 4D tensor

    if device is not None:
        epsilon = epsilon.to(device, non_blocking=pin_memory)

    for i in range(max_steps):
        with torch.enable_grad():
            outputs = model(adv_inputs)

            loss = criterion(outputs, labels)
            assert torch.all(torch.isfinite(loss)).item()

            grad_outputs = torch.ones(loss.shape)
            if device is not None:
                grad_outputs = grad_outputs.to(device, non_blocking=pin_memory)

            grads = torch.autograd.grad(loss,
                                        adv_inputs,
                                        grad_outputs=grad_outputs,
                                        only_inputs=True)[0]

            if norm == 'inf':
                update = step_size * grads.sign()
            elif norm == '2':
                update = step_size * grads

            perturbed_image = adv_inputs + update

            # project the perturbed_image back onto the norm-ball
            if norm == 'inf':
                perturbed_image = torch.max(torch.min(perturbed_image, inputs + epsilon), inputs - epsilon)
            elif norm == '2':
                # as the first dim is just channels, find norm of the 2D image in each channel dim
                norm_value = perturbed_image.view(perturbed_image.shape[0], -1).norm(p=2, dim=1)
                mask = norm_value <= epsilon # result dim = num_channels
                scaling_factor = norm_value
                scaling_factor[mask] = epsilon # update only channels whose norm value is more than epsilon

                perturbed_image = perturbed_image * (epsilon / scaling_factor.view(-1, 1, 1, 1)) # convert to 4D tensor

            perturbed_image = torch.clamp(perturbed_image, -1, 1) # re-normalize the image to range (-1,1)
            adv_inputs.data = perturbed_image


            if max_loss is None:
                max_loss = loss.clone()
                best_adversary = adv_inputs.clone()
            else:
                old_new_best = torch.argmax(torch.cat((max_loss, loss), dim=1), dim=1)
                best_adversary[old_new_best == 1, :, :, :] = adv_inputs[old_new_best == 1, :,:,:]
                max_loss = torch.max(max_loss, loss)

    # return the best adversarial sample generated (one with max loss so far)
    return best_adversary