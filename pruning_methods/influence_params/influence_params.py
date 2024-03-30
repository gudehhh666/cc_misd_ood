import torch
from torch.autograd import grad
from pathlib import Path
import logging


def grad_z(z, t, model, device='cuda:0'):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    # initialize
    if device:
        z, t = z.to(device), t.to(device)
    y = model(z)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    return list(grad(loss, params, create_graph=True))

def calc_loss(y, t, reduction='mean'):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = torch.nn.functional.log_softmax(y, dim=0)
    y_ = torch.nn.functional.log_softmax(y)
    loss = torch.nn.functional.nll_loss(
        y_, t, weight=None, reduction=reduction)
    
    # alternative: cross_entropy
    # loss = torch.nn.functional.cross_entropy(y, t, reduction='mean')
    return loss

def calc_influence(model, train_loader, save=False, device='cuda:0',
                damp=0.01, scale=25, recursion_depth=5000, r=1, start=0):
    
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")
    
    
    s_trains = []
    h_estimates = []
    
    for i in range(start, len(train_loader.dataset)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        z, t = z.to(device), t.to(device)
        s_train = grad_z(z, t, model, device=device)
        s_trains.append(s_train)
    
    for i in range(len(s_trains)):
        h_estimate = s_trains[i].copy()
        for i in range(recursion_depth):
            for x, t in train_loader:
                x, t = x.to(device), t.to(device)
                y = model(x)
                loss = calc_loss(y, t)
                
                
                params = [ p for p in model.parameters() if p.requires_grad ]
                hv = hvp(loss, params, h_estimate)
                
                h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
                break
        h_estimates.append(h_estimate)
                
            
    
    return s_trains, h_estimates


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads

