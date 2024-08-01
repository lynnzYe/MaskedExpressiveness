from maskexp.model.create_dataset import load_dataset
from maskexp.definitions import DATA_DIR, DEFAULT_MASK_EVENT
from maskexp.util.prepare_data import mask_perf_tokens, get_attention_mask
from maskexp.magenta.pipelines.performance_pipeline import get_full_token_pipeline
from maskexp.util.tokenize_midi import extract_complete_tokens
from tools import ExpConfig, load_model, load_model_from_pth, print_perf_seq, decode_batch_perf_logits, hits_at_k, \
    accuracy_within_n, bind_metric, decode_perf_logits
from maskexp.util.play_midi import syn_perfevent
from maskexp.magenta.models.performance_rnn import performance_model
from maskexp.model.bert import NanoBertMLM
from pathlib import Path
import torch
import functools
import numpy as np
import math
import matplotlib.pyplot as plt
import note_seq
import os
import tqdm
import time

NDEBUG = False


def test_mlm(model, perf_config, test_dataloader, metrics=None, device=torch.device('mps')):
    if metrics is None:
        metrics = [hits_at_k, accuracy_within_n]

    test_metrics = {metric.__name__: [] for metric in metrics}
    tokenizer = perf_config.encoder_decoder
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm.tqdm(test_dataloader)):
            input_ids = batch[0]
            attention_mask = batch[1]

            decoded = [tokenizer.class_index_to_event(i, None) for i in input_ids[0, :].tolist()]
            print("intpus:")
            print_perf_seq(decoded)
            # Mask tokens
            inputs, labels = mask_perf_tokens(input_ids, perf_config=perf_config, mask_prob=0.15,
                                              special_ids=(note_seq.PerformanceEvent.VELOCITY,))
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, logits = model(inputs, attention_mask=attention_mask, labels=labels)
            total_loss += loss.item()

            # Calculate and store each metric
            for metric in metrics:
                metric_value = metric(logits, labels)
                if not math.isnan(metric_value):
                    test_metrics[metric.__name__].append(metric_value)

            if step == len(test_dataloader) - 1 or not NDEBUG:
                decoded = [tokenizer.class_index_to_event(i, None) for i in inputs[0, :].tolist()]
                print("Input ids:")
                print_perf_seq(decoded)
                print("Decoded:")
                print_perf_seq(decode_batch_perf_logits(logits, tokenizer._one_hot_encoding, idx=0))

            if not NDEBUG:
                break

    avg_test_loss = total_loss / len(test_dataloader)
    print(f'Test Loss: {avg_test_loss}')

    # Average out metrics over all batches
    avg_test_metrics = {metric: sum(values) / len(values) for metric, values in test_metrics.items()}
    for metric_name, avg_value in avg_test_metrics.items():
        print(f'{metric_name}: {avg_value}')

    # Save the checkpoint
    return avg_test_loss, avg_test_metrics


def run_mlm_test(test_settings: ExpConfig = None):
    if test_settings is None:
        test_settings = ExpConfig()
    perf_config = performance_model.default_configs[test_settings.perf_config_name]

    _, _, test = load_dataset(test_settings.data_path)

    model = NanoBertMLM(vocab_size=perf_config.encoder_decoder.num_classes,
                        n_embed=test_settings.n_embed,
                        max_seq_len=test_settings.max_seq_len,
                        n_layers=test_settings.n_layers,
                        n_heads=test_settings.n_heads,
                        dropout=test_settings.dropout)
    model.to(test_settings.device)
    load_model(model, None, f'{test_settings.save_dir}/checkpoints/{test_settings.model_name}.pth')

    hits_1, hits_3, hits_5 = (bind_metric(hits_at_k, k=1),
                              bind_metric(hits_at_k, k=3),
                              bind_metric(hits_at_k, k=5))
    acc_wi_1, acc_wi_2, acc_wi_3 = (bind_metric(accuracy_within_n, n=1),
                                    bind_metric(accuracy_within_n, n=2),
                                    bind_metric(accuracy_within_n, n=3))

    avg_test_loss, metrics = test_mlm(model, perf_config, test,
                                      metrics=[hits_1, hits_3, hits_5, acc_wi_1, acc_wi_2, acc_wi_3]
                                      )
    metrics['loss'] = avg_test_loss

    torch.save(metrics,
               f'/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/logs/{test_settings.model_name}-metrics.pth')
    return


def test_velocitymlm(ckpt_path):
    cfg = ExpConfig.load_from_dict(torch.load(ckpt_path))
    run_mlm_test(test_settings=cfg)


def pad_seq(tokens, pad_id=0, seq_len=256):
    return torch.tensor(np.pad(tokens, (0, seq_len - len(tokens)), mode='constant', constant_values=pad_id))


def prepare_model_input(list_of_encodings, perf_config=None, seq_len=256):
    """
    convert a list of one_hot encodings to array of appropriately-sized sequences for model prediction
    :param list_of_encodings:
    :param perf_config:
    :return:
    """
    if perf_config is None:
        raise ValueError("need config")
    chunks = [torch.tensor(list_of_encodings[i:i + seq_len]) for i in range(0, len(list_of_encodings), seq_len)]
    masks = [torch.tensor(get_attention_mask(ck, max_seq_len=seq_len)) for ck in chunks]
    pad_token = perf_config.encoder_decoder.default_event_label
    if len(chunks[-1]) < seq_len:
        chunks[-1] = pad_seq(chunks[-1], pad_id=pad_token, seq_len=seq_len)

    return chunks, masks


def get_demo_token_mask(token_ids, one_hot_encoding,
                        special_token=note_seq.PerformanceEvent(event_type=4, event_value=1)):
    """
    Mask velocity=1 events for demo evaluation
    :param token_ids:
    :param one_hot_encoding:
    :param special_token:
    :return:
    """
    out = []
    for i in token_ids:
        event = one_hot_encoding.decode_event(i)
        if event.event_type == special_token.event_type and event.event_value == special_token.event_value:
            out.append(True)  # Replace with MASK TOKEN
        else:
            out.append(False)
    return out


def mask_velocity_demo_tokens(token_seq_list, perf_config=None):
    """
    Replace velocity=1 events with MSK event (which is defined in prepare_data mask_perf_token
    :param token_seq_list:
    :param perf_config
    :return:
    """
    if perf_config is None:
        raise ValueError("Config is required to use the tokenizer!")

    # Obtain special tokens
    tokenizer = perf_config.encoder_decoder._one_hot_encoding
    for seq in token_seq_list:
        mask = torch.tensor(get_demo_token_mask(seq, tokenizer))
        seq[mask] = tokenizer.encode_event(DEFAULT_MASK_EVENT)


def mlm_pred(model, perf_config, input_list, mask_list, device=torch.device('mps')):
    assert len(input_list) == len(mask_list) and len(input_list) > 0
    assert len(input_list[0]) == len(mask_list[0])
    assert isinstance(input_list[0], torch.Tensor)
    assert isinstance(mask_list[0], torch.Tensor)

    tokenizer = perf_config.encoder_decoder
    model.eval()

    out = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(input_list))):
            inputs = input_list[i].unsqueeze(0).to(device)
            attention_mask = mask_list[i].unsqueeze(0).to(device)
            _, logits = model(inputs, attention_mask=attention_mask, labels=None)
            out.extend(decode_perf_logits(logits, tokenizer._one_hot_encoding))
    return out


def run_mlm_pred(pth, inputs, masks):
    perf_config = performance_model.default_configs[pth['perf_config_name']]
    model = NanoBertMLM(vocab_size=perf_config.encoder_decoder.num_classes,
                        n_embed=pth['n_embed'],
                        max_seq_len=pth['max_seq_len'],
                        n_layers=pth['n_layers'],
                        n_heads=pth['n_heads'],
                        dropout=pth['dropout'])
    model.to(pth['device'])
    load_model_from_pth(model, optimizer=None, pth=pth)
    return mlm_pred(model, perf_config, inputs, masks, device=pth['device'])


def mask_all_velocity_ids(token_ids, perf_config):
    mask_id = perf_config.encoder_decoder._one_hot_encoding.encode_event(DEFAULT_MASK_EVENT)
    for i, e in enumerate(token_ids):
        event = perf_config.encoder_decoder._one_hot_encoding.decode_event(e)
        if event.event_type == note_seq.PerformanceEvent.VELOCITY:
            token_ids[i] = mask_id


def render_seq(midi_path, ckpt_path=None, mask_all_velocity=False):
    """
    Predict velocity for masked midi events
        -> those note-on velocity events to be predicted should use velocity = 1 -> converted to 4-0
        -> generation by the simplest strategy -> shift by max-seq-len, no window overlap
    :param midi_path:
    :param ckpt_path: path to the checkpoint (which should also contain the model settings, losses etc.)
    :return:
    """
    if ckpt_path is None:
        raise ValueError("Config path must be provided")
    pth = torch.load(ckpt_path)
    cfg = ExpConfig.load_from_dict(pth)
    perf_config = performance_model.default_configs[cfg.perf_config_name]
    tokens = extract_complete_tokens(midi_path, perf_config, max_seq=None)

    inputs, masks = prepare_model_input(tokens, perf_config, seq_len=pth['max_seq_len'])
    if mask_all_velocity:
        for ids in inputs:
            mask_all_velocity_ids(ids, perf_config)

    # Model Prediction
    out = run_mlm_pred(pth, inputs, masks)
    syn_perfevent(out, filename='demo1.wav', n_velocity_bin=perf_config.num_velocity_bins)
    pass


if __name__ == '__main__':
    test_velocitymlm('/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/velocitymlm++++++.pth')
    # render_seq('../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid',
    #            ckpt_path='/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/velocitymlm+++.pth',
    #            mask_all_velocity=True)
