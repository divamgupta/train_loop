import torch
from torch import nn
import torch.nn.functional as F
from train_loop.utils.dynamic_import import build_class, get_obj

class FunctionLossWrapper:
    """Wrapper to make function losses compatible with class-based interface."""
    def __init__(self, loss_fn, **kwargs):
        self.loss_fn = loss_fn
        self.kwargs = kwargs
    
    def __call__(self, batch, model_outs):
        return self.loss_fn(batch, model_outs, **self.kwargs)

def make_loss_module(loss_name, loss_args):
    args = {k: v for k, v in loss_args.items() if k not in ['weight', 'function_name']}

    if get_obj(loss_name, return_none_if_not_found=True) is None:
        # Try to get the loss function/class from globals
        loss_obj = globals().get(loss_name)
    else:
        loss_obj = get_obj(loss_name)
        print("Got  ext loss object:", loss_obj)
    
    if loss_obj is None:
        # Fallback to build_class for external losses
        config = {
            'name': loss_name,
            'args': args
        }
        return build_class(config)
    elif callable(loss_obj) and not isinstance(loss_obj, type):
        # It's a function, wrap it
        return FunctionLossWrapper(loss_obj, **args)
    else:
        return loss_obj(**args)


class LossModule():
    def __init__(self, **losses):
        self.loss_functions = {}
        self.loss_weights = {}
        for loss_name , loss_args in losses.items():
            self.loss_weights[loss_name] = loss_args.get('weight', 1.0)
            function_name = loss_args.get('function_name', loss_name)
            self.loss_functions[loss_name] = make_loss_module(function_name, loss_args)
            


    def __call__(self, batch , model_outs ):
        loss = 0.0
        loss_dict = {}
        for loss_name, loss_fn in self.loss_functions.items():
            
            if self.loss_weights[loss_name] > 0:
                loss_value = loss_fn(batch, model_outs)
                loss += self.loss_weights[loss_name] * loss_value
                loss_dict[loss_name] = loss_value 
          
        return loss, loss_dict


def get_target(tgt_key, batch, model_outs):
    """Return the target tensor from batch or model_outs, preferring batch."""
    if tgt_key in batch:
        return batch[tgt_key]
    return model_outs[tgt_key]

def mse(batch, model_outs, src_key , tgt_key):
    gt = get_target(tgt_key, batch, model_outs)
    pred = model_outs[src_key]
    return F.mse_loss(pred, gt)

def flexible_l1(batch, model_outs, src_key, tgt_key):
    """Flexible L1 loss that handles both single tensors and lists of tensors."""
    pred = model_outs[src_key]
    pred = model_outs[src_key]
    gt = get_target(tgt_key, batch, model_outs)
    if type(pred) is list:
        list_loss = 0
        for i in range(len(pred)):
            list_loss += F.l1_loss(pred[i], gt[i])
        return list_loss / len(pred)
    else:
        return F.l1_loss(pred, gt)

def flexible_l1_smooth(batch, model_outs, src_key, tgt_key):
    """Flexible L1 loss that handles both single tensors and lists of tensors."""
    pred = model_outs[src_key]
    gt = get_target(tgt_key, batch, model_outs)
    if type(pred) is list:
        list_loss = 0
        for i in range(len(pred)):
            list_loss += F.smooth_l1_loss(pred[i], gt[i])
        return list_loss / len(pred)
    else:
        return F.smooth_l1_loss(pred, gt)

def flexible_bce(batch, model_outs, src_key, tgt_key):
    """Flexible BCE loss that handles both single tensors and lists of tensors."""
    pred = model_outs[src_key]
    gt = get_target(tgt_key, batch, model_outs)
    if type(pred) is list:
        list_loss = 0
        for i in range(len(pred)):
            list_loss += F.binary_cross_entropy(pred[i], gt[i])
        return list_loss / len(pred)
    else:
        return F.binary_cross_entropy(pred, gt)

def cosine_loss(batch, model_outs, src_key, tgt_key, dim=-1):
    """Cosine similarity loss between predictions and ground truth. Dim is the dimension where you have the vectors"""
    pred = model_outs[src_key]
    gt = get_target(tgt_key, batch, model_outs)
    return (1 - F.cosine_similarity(pred, gt, dim=dim)).mean()

def cross_entropy_classification_loss(batch, model_outs, src_key, tgt_key):
    """
    Classification loss for logits and class IDs.
    Args:
        batch: dict containing ground truth class IDs (tgt_key)
        model_outs: dict containing predicted logits (src_key)
        src_key: key for logits in model_outs
        tgt_key: key for class IDs in batch or model_outs
    Returns:
        Cross-entropy loss
    """
    logits = model_outs[src_key]
    class_ids = get_target(tgt_key, batch, model_outs)
    return F.cross_entropy(logits, class_ids)

def accuracy_classification(batch, model_outs, src_key, tgt_key, top_k=1):
    """
    Accuracy metric for classification with top-k support.
    Args:
        batch: dict containing ground truth class IDs (tgt_key)
        model_outs: dict containing predicted logits (src_key)
        src_key: key for logits in model_outs
        tgt_key: key for class IDs in batch or model_outs
        top_k: int, number of top predictions to consider (default: 1)
               top_k=1 gives standard accuracy
               top_k=5 gives top-5 accuracy, etc.
    Returns:
        Accuracy as a float tensor
    """
    logits = model_outs[src_key]
    class_ids = get_target(tgt_key, batch, model_outs)
    
    if top_k == 1:
        # Standard accuracy (faster path)
        preds = torch.argmax(logits, dim=-1)
        return (preds == class_ids).float().mean()
    else:
        # Top-k accuracy
        _, top_k_preds = torch.topk(logits, k=top_k, dim=-1)
        # Expand class_ids to compare with each of the top-k predictions
        class_ids_expanded = class_ids.unsqueeze(-1).expand_as(top_k_preds)
        # Check if true class is in any of the top-k predictions
        correct = (top_k_preds == class_ids_expanded).any(dim=-1)
        return correct.float().mean()