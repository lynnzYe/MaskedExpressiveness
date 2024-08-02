import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import os
import note_seq
from inspect import signature
from maskexp.definitions import IGNORE_LABEL_INDEX, DEFAULT_MASK_EVENT, VELOCITY_MASK_EVENT
from pathlib import Path
from maskexp.magenta.models.performance_rnn import performance_model

MAX_SEQ_LEN = 128


def load_model(model, optimizer=None, cpath=''):
    ckpt_path = Path(cpath)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    load_model_from_pth(model, optimizer, ckpt)


def load_model_from_pth(model, optimizer=None, pth=None):
    if pth is None:
        raise ValueError(".pth file must be provided")
    model.load_state_dict(pth['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(pth['optimizer_state_dict'])


def decode_batch_perf_logits(logits, decoder=None, idx=None):
    if decoder is None:
        raise ValueError("Decoder is required")
    if idx is None:
        pred_ids = torch.argmax(F.softmax(logits, dim=-1), dim=-1).tolist()
    else:
        pred_ids = torch.argmax(F.softmax(logits, dim=-1), dim=-1)[idx, :].tolist()
    out = []
    for tk in pred_ids:
        out.append(decoder.decode_event(tk))
    return out


def decode_perf_logits(logits, decoder=None):
    if decoder is None:
        raise ValueError("Decoder is required")
    pred_ids = torch.argmax(F.softmax(logits, dim=-1), dim=-1).tolist()
    out = []
    for tk in pred_ids[0]:
        out.append(decoder.decode_event(tk))
    return out


def compare_perf_event(e, target_event):
    if e.event_value == target_event.event_value and e.event_type == target_event.event_type:
        return True
    return False


def print_perf_seq(perf_seq):
    outstr = ''
    for e in perf_seq:
        if compare_perf_event(e, DEFAULT_MASK_EVENT):
            outstr += f'\x1B[34m[{e.event_type}-{e.event_value}],\033[0m\t'
        elif compare_perf_event(e, VELOCITY_MASK_EVENT):
            outstr += f'\x1B[33m[{e.event_type}-{e.event_value}],\033[0m\t'
        else:
            outstr += f'[{e.event_type}-{e.event_value}],\t'
    print(outstr)


def cross_entropy_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL_INDEX)
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item()


def hits_at_k(logits, labels, k=3, ignore_index=IGNORE_LABEL_INDEX):
    """
    HITS@K metric
    :param logits:
    :param labels:
    :param k:
    :param ignore_index:
    :return:
    """
    _, topk_indices = torch.topk(logits, k, dim=-1)
    valid_mask = labels != ignore_index  # obtain boolean tensor for masking
    valid_labels = labels[valid_mask]
    valid_topk_indices = topk_indices[valid_mask]

    if valid_labels.numel() == 0:
        return float('nan')

    # .mean() effectively calculates the binary hit/miss values
    hits = (valid_topk_indices == valid_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return hits


def accuracy_within_n(logits, labels, n=1, ignore_index=IGNORE_LABEL_INDEX):
    """
    Calculate accuracy ± n for ordinal classification
    :param logits:
    :param labels:
    :param n:
    :param ignore_index:
    :return:
    """
    pred = logits.argmax(dim=-1)
    valid_mask = labels != ignore_index
    valid_pred = pred[valid_mask]
    valid_labels = labels[valid_mask]
    if valid_labels.numel() == 0:
        return float('nan')
    acc = (torch.abs(valid_pred - valid_labels) <= n).float().mean().item()
    return acc


def bind_metric(func, **kwargs):
    partial_func = functools.partial(func, **kwargs)
    arg_str = "_".join(f"{key}{value}" for key, value in kwargs.items())
    partial_func.__name__ = f"{func.__name__}_{arg_str}"
    return partial_func


class ExpConfig:
    def __init__(self, model_name='', save_dir='', data_path='', perf_config_name='',
                 n_embed=256, n_layers=4, n_heads=4, dropout=0.1, max_seq_len=MAX_SEQ_LEN, special_tokens=None,
                 device=torch.device('mps'), n_epochs=20, mlm_prob=0.15, lr=1e-4, eval_interval=5, save_interval=5,
                 resume_from=None, train_loss=None, val_loss=None, epoch=0):
        assert os.path.exists(save_dir) and os.path.exists(data_path)
        ckpt_save_path = os.path.join(save_dir, model_name, '.pth')
        if os.path.exists(ckpt_save_path):
            raise FileExistsError(f"found existing checkpoint file at {ckpt_save_path}")
        if perf_config_name not in performance_model.default_configs.keys():
            raise KeyError(f"Performance config key: {perf_config_name} not found")
        if special_tokens is None:
            print("\x1B[33m[Warning]\033[0m Special token(s) is required for MLM training")

        # IO Paths
        self.model_name = model_name  # Will be used to name the saved file
        self.save_dir = save_dir  # two folders will be created - checkpoints, logs
        self.data_path = data_path
        self.perf_config_name = perf_config_name

        # Model Setting
        self.n_embed = n_embed
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        # Training Setting
        self.lr = lr
        self.mlm_prob = mlm_prob
        self.device = device
        self.n_epochs = n_epochs
        self.eval_intv = eval_interval
        self.save_intv = save_interval
        self.special_tokens = special_tokens

        self.resume_from = resume_from  # Provide checkpoint path to resume

        self.train_loss = [] if train_loss is None else train_loss
        self.val_loss = [] if val_loss is None else val_loss
        self.epoch = epoch

    @classmethod
    def load_from_dict(cls, json_cfg):
        init_params = signature(cls.__init__).parameters
        filtered_cfg = {key: value for key, value in json_cfg.items() if key in init_params}
        return cls(**filtered_cfg)
