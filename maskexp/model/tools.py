import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import os
import note_seq
from maskexp.definitions import IGNORE_LABEL_INDEX
from pathlib import Path
from maskexp.magenta.models.performance_rnn import performance_model

MAX_SEQ_LEN = 128


def load_ckpt(model, optimizer=None, cpath=''):
    ckpt_path = Path(cpath)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])


def decode_perf_logits(logits, decoder=None, idx=0):
    if decoder is None:
        raise ValueError("Decoder is required")
    pred_ids = torch.argmax(F.softmax(logits, dim=-1), dim=-1)[0, :].tolist()
    out = []
    for tk in pred_ids:
        out.append(decoder.decode_event(tk))
    return out


def print_perf_seq(perf_seq):
    outstr = ''
    for e in perf_seq:
        outstr += f'[{e.event_type}-{e.event_value}], '
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
                 device=torch.device('mps'), n_epochs=20, mlm_prob=0.15, eval_interval=5, resume_from=None):

        assert os.path.exists(save_dir) and os.path.exists(data_path)
        ckpt_save_path = os.path.join(save_dir, model_name, '.pth')
        if os.path.exists(ckpt_save_path):
            raise FileExistsError(f"found existing checkpoint file at {ckpt_save_path}")
        if perf_config_name not in performance_model.default_configs.keys():
            raise KeyError(f"Performance config key: {perf_config_name} not found")
        if special_tokens is None:
            raise ValueError("Special token(s) is required for MLM training")

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
        self.mlm_prob = mlm_prob
        self.device = device
        self.n_epochs = n_epochs
        self.eval_intv = eval_interval
        self.special_tokens = special_tokens

        self.resume_from = resume_from  # Provide checkpoint path to resume
