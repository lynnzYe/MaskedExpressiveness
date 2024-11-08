import functools
import json
import math
import random
import os

import note_seq
import numpy as np
import torch
import tqdm
from sklearn.metrics import cohen_kappa_score
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from maskexp.util.play_midi import syn_perfevent, decode_output_ids, write_ids_to_midi, write_token_to_midi
from maskexp.definitions import OUTPUT_DIR, SAVE_DIR, VELOCITY_MASK_EVENT, NDEBUG
from maskexp.magenta.models.performance_rnn import performance_model
from maskexp.model.bert import NanoBertMLM
from maskexp.model.create_dataset import load_dataset
from maskexp.util.play_midi import performance_events_to_pretty_midi
from maskexp.util.prepare_data import mask_perf_tokens, get_attention_mask
from maskexp.util.tokenize_midi import extract_complete_tokens, midi_cleanup
from maskexp.model.tools import ExpConfig, load_model, load_model_from_pth, print_perf_seq, hits_at_k, \
    accuracy_within_n, bind_metric, logits_to_id, load_torch_model


def test_mlm(model, perf_config, test_dataloader, metrics=None, device=torch.device('mps')):
    if metrics is None:
        metrics = [hits_at_k, accuracy_within_n]

    test_metrics = {metric.__name__: [] for metric in metrics}
    tokenizer = perf_config.encoder_decoder
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            input_ids = batch[0]
            attention_mask = batch[1]

            # Mask tokens
            inputs, labels, mask_type_tensor = mask_perf_tokens(input_ids, perf_config=perf_config, mask_prob=0.15,
                                                                # normal_mask_ratio=.3,
                                                                special_ids=(note_seq.PerformanceEvent.VELOCITY,))
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss, logits = model(inputs, attention_mask=attention_mask, labels=labels)
            total_loss += loss.item()
            count += 1

            # Calculate and store each metric
            for metric in metrics:
                metric_value = metric(logits, labels, masks=mask_type_tensor)
                if not math.isnan(metric_value):
                    test_metrics[metric.__name__].append(metric_value)

            # if step == len(test_dataloader) - 1 or not NDEBUG:
            #     decoded = [tokenizer.class_index_to_event(i, None) for i in inputs[0, :].tolist()]
            #     print("Input ids:")
            #     print_perf_seq(decoded)
            #     print("Decoded:")
            #     print_perf_seq(decode_batch_perf_logits(logits, tokenizer._one_hot_encoding, idx=0))

            if not NDEBUG:
                break

    avg_test_loss = total_loss / count
    # print(f'Test Loss: {avg_test_loss}')

    # Average out metrics over all batches
    avg_test_metrics = {metric: sum(values) / len(values) for metric, values in test_metrics.items()}
    # for metric_name, avg_value in avg_test_metrics.items():
    #     print(f'{metric_name}: {avg_value}')

    # Save the checkpoint
    return avg_test_loss, avg_test_metrics


def get_mlm_metrics(test_settings: ExpConfig = None):
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

    hits_1_all, hits_2_all, hits_3_all = (bind_metric(hits_at_k, k=1, consider_mask=torch.tensor([1, 2])),
                                          bind_metric(hits_at_k, k=2, consider_mask=torch.tensor([1, 2])),
                                          bind_metric(hits_at_k, k=3, consider_mask=torch.tensor([1, 2])))
    hits_1, hits_2, hits_3 = (bind_metric(hits_at_k, k=1, consider_mask=torch.tensor([1])),
                              bind_metric(hits_at_k, k=2, consider_mask=torch.tensor([1])),
                              bind_metric(hits_at_k, k=3, consider_mask=torch.tensor([1])))
    acc_w_1_all, acc_w_2_all, acc_w_3_all = (bind_metric(accuracy_within_n, n=1, consider_mask=torch.tensor([1, 2])),
                                             bind_metric(accuracy_within_n, n=2, consider_mask=torch.tensor([1, 2])),
                                             bind_metric(accuracy_within_n, n=3, consider_mask=torch.tensor([1, 2])))
    acc_w_1, acc_w_2, acc_w_3 = (bind_metric(accuracy_within_n, n=1, consider_mask=torch.tensor([1])),
                                 bind_metric(accuracy_within_n, n=2, consider_mask=torch.tensor([1])),
                                 bind_metric(accuracy_within_n, n=3, consider_mask=torch.tensor([1])))

    avg_test_loss, metrics = test_mlm(model, perf_config, test,
                                      metrics=[
                                          hits_1, hits_2, hits_3, hits_1_all, hits_2_all, hits_3_all,
                                          acc_w_1, acc_w_2, acc_w_3, acc_w_1_all, acc_w_2_all, acc_w_3_all
                                      ]
                                      )
    metrics['loss'] = avg_test_loss
    return metrics


def run_mlm_test(ckpt_path, test_times=100):
    cfg = ExpConfig.load_from_dict(load_torch_model(ckpt_path))
    metrics = []
    for i in tqdm.tqdm(range(test_times)):
        metrics.append(get_mlm_metrics(test_settings=cfg))
    avg_metric = {key: 0 for key in metrics[0].keys()}
    for e in metrics:
        for name, val in e.items():
            avg_metric[name] += val
    for name in avg_metric.keys():
        avg_metric[name] /= test_times

    print(json.dumps(avg_metric, indent=4))
    torch.save(avg_metric,
               f'/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/logs/{cfg.model_name}-metrics.pth')
    return avg_metric


def pad_seq(tokens, pad_id=0, seq_len=256):
    return torch.tensor(np.pad(tokens, (0, seq_len - len(tokens)), mode='constant', constant_values=pad_id))


def prepare_model_input(list_of_encodings, perf_config=None, seq_len=128):
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


def prepare_contextual_model_input(list_of_encodings, perf_config=None, seq_len=128,
                                   overlap=0.5):
    """
    convert a list of one_hot encodings to array of appropriately-sized sequences (with overlap) for model prediction
    :param list_of_encodings:
    :param perf_config:
    :param seq_len:
    :param overlap:
    :return:
    """
    if perf_config is None:
        raise ValueError("need config")
    # Calculate the stride based on the overlap
    stride = int(seq_len * (1 - overlap))

    # Create chunks with overlap
    chunks = [torch.tensor(list_of_encodings[i:i + seq_len]) for i in
              range(0, len(list_of_encodings) - stride, stride)]

    pad_token = perf_config.encoder_decoder.default_event_label
    # Create attention masks for each chunk
    masks = [torch.tensor(get_attention_mask(ck, max_seq_len=seq_len)) for ck in chunks]

    # Pad the last chunk if it's shorter than seq_len
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
        seq[mask] = tokenizer.encode_event(VELOCITY_MASK_EVENT)


def mlm_pred(model, perf_config, input_list, mask_list, device=torch.device('mps'), replace_id=2):
    """
    Perform prediction on masked input sequence
    :param model:
    :param perf_config:
    :param input_list:
    :param mask_list:
    :param device:
    :return:
    """
    assert len(input_list) == len(mask_list) and len(input_list) > 0
    assert len(input_list[0]) == len(mask_list[0])
    assert isinstance(input_list[0], torch.Tensor)
    assert isinstance(mask_list[0], torch.Tensor)

    tokenizer = perf_config.encoder_decoder
    model.eval()

    out = []
    with torch.no_grad():
        for i in range(len(input_list)):
            input_ids = input_list[i].to(device)
            replace_ids = input_ids == replace_id
            idx_replace = replace_ids.nonzero(as_tuple=True)[0]
            if len(idx_replace) == 0:
                continue

            attention_mask = mask_list[i].unsqueeze(0).to(device)
            _, logits = model(input_ids.unsqueeze(0).to(device), attention_mask=attention_mask, labels=None)
            pred_ids = logits_to_id(logits).to(device)

            out_ids = input_ids.clone()
            out_ids[idx_replace] = pred_ids[idx_replace]
            out.extend(decode_output_ids(out_ids, tokenizer._one_hot_encoding))

    if NDEBUG:
        input_decoded = [tokenizer.class_index_to_event(i, None) for i in input_list[0].tolist()]
        output_decoded = out[:len(input_decoded)]
        print("Input 128:")
        print_perf_seq(input_decoded)
        print("Output 128:")
        print_perf_seq(output_decoded)
    return out


def step_contextual_velocity_mlm_pred(model, input_ids, mask, start_index=0, num_to_replace=0, overlap=0.5,
                                      device=torch.device('mps'), replace_id=2):
    """
    Predict and replace % of the special token
    :param model:
    :param input_ids:
        masked tokens are replaced with VELOCITY_MASK_EVENT
        cover_percent of them will be replaced per call
    :param mask:
    :param num_to_replace:
    :param overlap:
    :param device:
    :param replace_id:
    :return: updated input_ids, and a bool status whether all are replaced (updated)
    """
    assert len(input_ids) == len(mask)
    if num_to_replace == 0:
        print("\x1B[33m[Warning]\033[0m no replacement asked. returning")
        return input_ids.de, False
    model.eval()

    with torch.no_grad():
        replace_ids = input_ids == replace_id
        available_indices = replace_ids.nonzero(as_tuple=True)[0]
        replaceable_indices = available_indices[available_indices >= start_index]
        if len(replaceable_indices) == 0:
            # print("\x1B[34m[Info]\033[0m no more replaceable indices!")
            return input_ids, False

        input_ids = input_ids.to(device)
        mask = mask.to(device)
        _, logits = model(input_ids.unsqueeze(0), attention_mask=mask.unsqueeze(0), labels=None)
        output_ids = logits_to_id(logits).to(device)

        # Randomly choose which indices to replace
        indices_to_replace = torch.randperm(len(replaceable_indices))[:num_to_replace]

        # Replace selected tokens with predicted tokens
        input_ids[replaceable_indices[indices_to_replace]] = output_ids[
            replaceable_indices[indices_to_replace]]

    return input_ids, True


def run_mlm_pred_full(pth, inputs, masks):
    """
    Predict all the masked events at once
    :param pth:
    :param inputs:
    :param masks:
    :return:
    """
    perf_config = performance_model.default_configs[pth['perf_config_name']]

    model = NanoBertMLM(vocab_size=perf_config.encoder_decoder.num_classes,
                        n_embed=pth['n_embed'],
                        max_seq_len=pth['max_seq_len'],
                        n_layers=pth['n_layers'],
                        n_heads=pth['n_heads'],
                        dropout=pth['dropout'])
    model.to(pth['device'])
    model.to(pth['device'])
    load_model_from_pth(model, optimizer=None, pth=pth)
    return mlm_pred(model, perf_config, inputs, masks, device=pth['device'])


def get_overlap_length(seq_len, overlap):
    return seq_len - int(seq_len * overlap)


def decode_overlapped_ids(out_id_list, decoder, overlap=0.5):
    """
    Decode class id into performance events, but input should be list of overlapped ids list
    :param out_id_list:
    :param overlap:
    :return:
    """
    assert isinstance(out_id_list[0][0], torch.Tensor)
    decoded = []
    seq_len = len(out_id_list[0])
    overlap_end_idx = get_overlap_length(seq_len, overlap)
    for i, ids in enumerate(out_id_list):
        if i == 0:
            decoded.extend([decoder.decode_event(e.item()) for e in ids])
        else:
            decoded.extend([decoder.decode_event(e.item()) for e in ids[overlap_end_idx:]])
    return decoded


def run_contextual_mlm_pred(pth, inputs, masks, overlap=0.5, step_percent=0.1):
    """
    Sliding window + step-by-step generation
    :param pth: Configuration dictionary containing model and device settings
    :param inputs: List of input tensors
    :param masks: List of attention masks corresponding to the inputs
    :param overlap: [warn] TODO not checked if it passes through all function calls!
            represents the Fraction of overlap between consecutive input sequences
    :param step_percent: Percentage of masked tokens to replace in each step
    :return: None
    """
    assert 0 < step_percent <= 1
    perf_config = performance_model.default_configs[pth['perf_config_name']]

    model = NanoBertMLM(vocab_size=perf_config.encoder_decoder.num_classes,
                        n_embed=pth['n_embed'],
                        max_seq_len=pth['max_seq_len'],
                        n_layers=pth['n_layers'],
                        n_heads=pth['n_heads'],
                        dropout=pth['dropout'])
    model.to(pth['device'])
    load_model_from_pth(model, optimizer=None, pth=pth)

    seq_len = inputs[0].size(0)
    # velocity, note-on, note-off, time-shifts. Velocity occupies approximately 1/4 of the inputs
    approx_velocity_event_count = int(seq_len / 4)
    num_pred_per_step = int(approx_velocity_event_count * step_percent)
    init_num_pred = int(num_pred_per_step / (1 - overlap))  # The more overlap, the more it should predict
    n_iterations = int((approx_velocity_event_count * (1 - overlap)) // num_pred_per_step + 1)
    overlap_length = get_overlap_length(seq_len, overlap)
    decoder = perf_config.encoder_decoder._one_hot_encoding
    replace_id = decoder.encode_event(VELOCITY_MASK_EVENT)

    inputs = [e.to(pth['device']) for e in inputs]
    masks = [e.to(pth['device']) for e in masks]

    while True:
        round_status = []
        for i_input, (input_ids, mask) in enumerate(zip(inputs, masks)):
            # TODO @Bmois remove debug code
            test_ids = input_ids.clone()
            pred_ids, status = step_contextual_velocity_mlm_pred(model, input_ids, mask,
                                                                 start_index=0 if i_input == 0 else overlap_length,
                                                                 overlap=overlap,
                                                                 num_to_replace=init_num_pred if i_input == 0 else num_pred_per_step,
                                                                 device=pth['device'],
                                                                 replace_id=replace_id)
            round_status.append(status)
            # indices = torch.where(input_ids != pred_ids)[0]
            # print(f"After modification: {i_input}", [(e.item(), pred_ids[e].item()) for e in indices])
            inputs[i_input] = pred_ids
            if i_input < len(inputs) - 1:
                # update the next input ids % percent with updated current input_ids
                inputs[i_input + 1][:overlap_length] = pred_ids[overlap_length:]

            # TODO @Bmois Debug
            write_ids_to_midi(inputs[i_input], perf_config)
            # test_tok = decode_overlapped_ids(inputs, decoder, overlap)
            # write_token_to_midi(test_tok, perf_config)

        if all(x == False for x in round_status):
            break

    for i, e in enumerate(inputs):
        if torch.any(e == 2):
            print(f"\x1B[33m[Warning]\033[0m Input No.{i} still have {torch.sum(e == 2).item()} velocity masks")

    perf_tokens = decode_overlapped_ids(inputs, decoder, overlap=overlap)
    return perf_tokens


def mask_all_velocity_ids(token_ids, perf_config):
    mask_id = perf_config.encoder_decoder._one_hot_encoding.encode_event(VELOCITY_MASK_EVENT)
    for i, e in enumerate(token_ids):
        event = perf_config.encoder_decoder._one_hot_encoding.decode_event(e)
        if event.event_type == note_seq.PerformanceEvent.VELOCITY:
            token_ids[i] = mask_id

    return token_ids


def mask_some_velocity_ids(token_ids, perf_config, mask_prob=0.9):
    mask_id = perf_config.encoder_decoder._one_hot_encoding.encode_event(VELOCITY_MASK_EVENT)
    count = 1
    for i, e in enumerate(token_ids):
        event = perf_config.encoder_decoder._one_hot_encoding.decode_event(e)
        if event.event_type == note_seq.PerformanceEvent.VELOCITY:
            count += 1
            prob = random.random()
            if prob < mask_prob:
                token_ids[i] = mask_id
    return token_ids


def mask_min_velocity_ids(token_ids, perf_config):
    mask_id = perf_config.encoder_decoder._one_hot_encoding.encode_event(VELOCITY_MASK_EVENT)
    for i, e in enumerate(token_ids):
        event = perf_config.encoder_decoder._one_hot_encoding.decode_event(e)
        if event.event_type == note_seq.PerformanceEvent.VELOCITY and event.event_value == 1:
            token_ids[i] = mask_id
    return token_ids


def mask_tokens(tokens, perf_config, mask_mode):
    """
    :param tokens:
    :param perf_config:
    :param mask_model:
        - all: ask all velocity tokens
        - some: randomly mask some velocity tokens
        - min: mask velocity tokens with value == 1
        - none: do nothing
    :return:
    """
    if mask_mode == 'all':
        return mask_all_velocity_ids(tokens, perf_config)
    elif mask_mode == 'some':
        return mask_some_velocity_ids(tokens, perf_config)
    elif mask_mode == 'min':
        return mask_min_velocity_ids(tokens, perf_config)
    elif mask_mode == 'none':
        return tokens
    else:
        raise ValueError("Mode incorrect")


def render_seq(midi_path, ckpt_path=None, mask_mode='all', file_stem='demo1', output_path=None):
    """
    Predict velocity for masked midi events
        -> those note-on velocity events to be predicted should use velocity = 1 -> converted to 4-0
        -> generation by the simplest strategy -> shift by max-seq-len, no window overlap
    :param midi_path: this might be overwritten!!
    :param ckpt_path: path to the checkpoint (which should also contain the model settings, losses etc.)
    :param mask_mode:
    :return:
    """
    if ckpt_path is None:
        raise ValueError("Config path must be provided")
    pth = load_torch_model(ckpt_path)
    cfg = ExpConfig.load_from_dict(pth)
    perf_config = performance_model.default_configs[cfg.perf_config_name]
    # clean_midi_path = os.path.join(OUTPUT_DIR, 'tmp_' + os.path.basename(midi_path))
    # midi_cleanup(midi_path, clean_midi_path)
    tokens = extract_complete_tokens(midi_path, perf_config, max_seq=None)
    # tokens = extract_complete_tokens(clean_midi_path, perf_config, max_seq=None)
    tokens = mask_tokens(tokens, perf_config=perf_config, mask_mode=mask_mode)
    if not NDEBUG:
        masked_perf = decode_output_ids(tokens, perf_config.encoder_decoder._one_hot_encoding)
        masked_mid = performance_events_to_pretty_midi(masked_perf)
        masked_mid.write(OUTPUT_DIR + "/masked_mid.mid")

    inputs, masks = prepare_model_input(tokens, perf_config, seq_len=pth['max_seq_len'])
    # Model Prediction
    out = run_mlm_pred_full(pth, inputs, masks)
    midi = performance_events_to_pretty_midi(out, perf_config.num_velocity_bins)
    if output_path is None:
        midi.write(f'{OUTPUT_DIR}/{file_stem}.mid')
    else:
        midi.write(os.path.join(output_path, file_stem + ".mid"))
    # syn_perfevent(out, filename='demo1.wav', n_velocity_bin=perf_config.num_velocity_bins)
    pass


def write_midi_chunk(chunks, config, overlap=0.5, stem='debug'):
    decoder = config.encoder_decoder._one_hot_encoding
    for i, e in enumerate(chunks):
        perf_tokens = decode_output_ids(e, decoder)
        mid = performance_events_to_pretty_midi(perf_tokens, config.num_velocity_bins)
        mid.write(OUTPUT_DIR + f'/{stem}-{i}.mid')


def render_contextual_seq(midi_path, ckpt_path=None, mask_mode='all', overlap=.5, file_stem='demo', output_path=None):
    """
    Predict velocity for masked midi events
        -> those note-on velocity events to be predicted should use velocity = 1 -> converted to 4-0
        -> generation by the simplest strategy -> shift by max-seq-len, no window overlap
    :param midi_path:
    :param mask_mode:
    :param ckpt_path: path to the checkpoint (which should also contain the model settings, losses etc.)
    :param overlap:
    :param file_stem:
    :param output_path:
    :return:
    """
    if ckpt_path is None:
        raise ValueError("Config path must be provided")
    pth = load_torch_model(ckpt_path)
    cfg = ExpConfig.load_from_dict(pth)
    perf_config = performance_model.default_configs[cfg.perf_config_name]
    tokens = extract_complete_tokens(midi_path, perf_config, max_seq=None)
    tokens = mask_tokens(tokens, perf_config=perf_config, mask_mode=mask_mode)
    overlapped_inputs, masks = prepare_contextual_model_input(tokens, perf_config, seq_len=pth['max_seq_len'],
                                                              overlap=.5)
    if not NDEBUG:
        masked_input = decode_overlapped_ids(overlapped_inputs, perf_config.encoder_decoder._one_hot_encoding,
                                             overlap=overlap)
        masked_mid = performance_events_to_pretty_midi(masked_input,
                                                       steps_per_second=perf_config.steps_per_second,
                                                       n_velocity_bin=perf_config.num_velocity_bins)
        masked_mid.write(f'{OUTPUT_DIR}/{file_stem}-input.mid')

    # Model Prediction
    out = run_contextual_mlm_pred(pth, overlapped_inputs, masks, overlap=overlap)

    midi = performance_events_to_pretty_midi(out, perf_config.num_velocity_bins)

    if output_path is None:
        midi.write(f'{OUTPUT_DIR}/{file_stem}.mid')
    else:
        midi.write(os.path.join(output_path, file_stem + ".mid"))
    # syn_perfevent(out, filename=f'{file_stem}.wav', n_velocity_bin=perf_config.num_velocity_bins)
    pass


def convert_kaggle_mlm(pth):
    data = load_torch_model(pth)
    data['device'] = torch.device('mps')
    data['save_dir'] = SAVE_DIR
    data['data_path'] = '/Users/kurono/Documents/python/GEC/ExpressiveMLM/data/mstro_with_dyn.pt'
    torch.save(data, f'{SAVE_DIR}/checkpoints/kg_{data["model_name"]}.pth')


def contextual_pred(tokens, pth, perf_config, overlap=0.5, mask_all_velocity=False):
    """
    input single list all tokens from a midi, output pred midi (velocity events updated only)
    :param tokens:
    :param pth:
    :param perf_config:
    :param seq_len:
    :param overlap:
    :param mask_all_velocity:
    :return:
    """
    in_decoded = [perf_config.encoder_decoder.class_index_to_event(i, None)
                  for i in tokens]
    tokens = mask_all_velocity_ids(tokens, perf_config) if mask_all_velocity \
        else mask_some_velocity_ids(tokens, perf_config, mask_prob=0.9)
    overlapped_inputs, masks = prepare_contextual_model_input(tokens, perf_config, seq_len=pth['max_seq_len'],
                                                              overlap=overlap)

    # Model Prediction
    out = run_contextual_mlm_pred(pth, overlapped_inputs, masks, overlap=overlap)
    return in_decoded, out


def raw_pred(tokens: list, pth, perf_config, mask_all_velocity=False):
    in_decoded = [perf_config.encoder_decoder.class_index_to_event(i, None)
                  for i in tokens]
    tokens = mask_all_velocity_ids(tokens, perf_config) if mask_all_velocity \
        else mask_some_velocity_ids(tokens, perf_config, mask_prob=0.9)
    inputs, masks = prepare_model_input(tokens, perf_config, seq_len=pth['max_seq_len'])

    # Model Prediction
    out = run_mlm_pred_full(pth, inputs, masks)
    return in_decoded, out


def eval_full_midi(midi_path, tokenizer, render_func, metric):
    """
    :param midi_path:
    :param tokenizer: func convert midi to tokens
    :param cfg: model config
    :param render_func: function that takes a token list (will be masked) as input
    :param metric: measure the distance between two list of tokens
    :return:
    """
    tokens = tokenizer(midi_path)
    decoded_input, result = render_func(tokens)
    stats = metric(decoded_input, result)
    return stats


def filter_full_midi_preds(perf_inputs: list[note_seq.PerformanceEvent], perf_results: list[note_seq.PerformanceEvent],
                           type_filter=(note_seq.PerformanceEvent.VELOCITY,)):
    """
    Obtain list of ytrue and ypred velocity values
    :param ytrue:
    :param ypred:
    :param type_filter:
    :return:
    """
    assert len(perf_inputs) <= len(perf_results)
    input_val = []
    output_val = []
    for i in range(len(perf_inputs)):
        pin = perf_inputs[i]
        pres = perf_results[i]
        if pin.event_type in type_filter or pres.event_type in type_filter:
            if pin.event_type != pres.event_type:
                print(f"\x1B[33m[Warning]\033[0m mismatched masking type in:{pin}-out:{pres} at idx {i}")
                # continue
            input_val.append(pin.event_value)
            output_val.append(pres.event_value)
    return input_val, output_val


def full_midi_qwk(perf_inputs: list[note_seq.PerformanceEvent], perf_results: list[note_seq.PerformanceEvent],
                  type_filter=(note_seq.PerformanceEvent.VELOCITY,)):
    """
    Quadratic Weighted Kappa (measures the agreement between two ratings)
    :param perf_inputs:
    :param perf_results:
    :param type_filter:
    :return:
    """
    ytrue, ypred = filter_full_midi_preds(perf_inputs, perf_results, type_filter=type_filter)
    return cohen_kappa_score(ytrue, ypred, weights='quadratic')


def full_midi_huber_loss(perf_inputs: list[note_seq.PerformanceEvent], perf_results: list[note_seq.PerformanceEvent],
                         type_filter=(note_seq.PerformanceEvent.VELOCITY,), delta=2):
    """
    Calculate the Huber loss between y_true and y_pred
    :param perf_inputs:
    :param perf_results:
    :param type_filter: default to velocity event
    :param delta:       3 velocity bins - default total 32 velocity bins. ±3 is like a 10-class classification
    :return:
    """
    ytrue, ypred = filter_full_midi_preds(perf_inputs, perf_results, type_filter=type_filter)
    error = np.array(ytrue) - np.array(ypred)
    is_small_error = np.abs(error) <= delta
    small_error_loss = 0.5 * error ** 2
    large_error_loss = delta * (np.abs(error) - 0.5 * delta)
    loss = np.where(is_small_error, small_error_loss, large_error_loss)
    return np.mean(loss)


def full_midi_dtw(perf_inputs: list[note_seq.PerformanceEvent], perf_results: list[note_seq.PerformanceEvent],
                  type_filter=(note_seq.PerformanceEvent.VELOCITY,)):
    ytrue, ypred = filter_full_midi_preds(perf_inputs, perf_results, type_filter=type_filter)
    distance, _ = fastdtw(ytrue, ypred, radius=len(ytrue), dist=3)
    return distance / len(ytrue)


def full_midi_acc_in_n(perf_inputs: list[note_seq.PerformanceEvent], perf_results: list[note_seq.PerformanceEvent],
                       type_filter=(note_seq.PerformanceEvent.VELOCITY,), n=3):
    ytrue, ypred = filter_full_midi_preds(perf_inputs, perf_results, type_filter=type_filter)
    ytrue = np.array(ytrue)
    ypred = np.array(ypred)
    differences = np.abs(ytrue - ypred)
    count_within_n = np.sum(differences <= n)

    accuracy = count_within_n / len(ytrue)
    return accuracy


def get_full_midi_metric():
    # metric = full_midi_qwk
    # metric = full_midi_huber_loss
    # metric = full_midi_dtw
    metric = full_midi_acc_in_n
    return metric


def get_context_funcs(ckpt_path, mask_all=False, overlap=0.5):
    pth = load_torch_model(ckpt_path)
    cfg = ExpConfig.load_from_dict(pth)
    perf_config = performance_model.default_configs[cfg.perf_config_name]
    tokenizer = functools.partial(extract_complete_tokens, config=perf_config, max_seq=None)

    render_func = functools.partial(contextual_pred, pth=pth, perf_config=perf_config,
                                    overlap=overlap, mask_all_velocity=mask_all)
    metric = get_full_midi_metric()

    return tokenizer, render_func, metric


def get_raw_funcs(ckpt_path, mask_all=False):
    pth = load_torch_model(ckpt_path)
    cfg = ExpConfig.load_from_dict(pth)
    perf_config = performance_model.default_configs[cfg.perf_config_name]
    tokenizer = functools.partial(extract_complete_tokens, config=perf_config, max_seq=None)

    render_func = functools.partial(raw_pred, pth=pth, perf_config=perf_config, mask_all_velocity=mask_all)
    metric = get_full_midi_metric()

    return tokenizer, render_func, metric


def test_eval_full_midi(midi_path, ckpt_path, mask_all=False, overlap=0.5, n_trials=500, mode='raw'):
    """
    The root function to be modified for testing generated midi
    :param midi_path:
    :return:
    """
    print("Mode:", mode)
    if mode == 'raw':
        funcs = get_raw_funcs(ckpt_path, mask_all=mask_all)
    else:
        funcs = get_context_funcs(ckpt_path, mask_all=mask_all, overlap=overlap)
    scores = []
    for i in tqdm.tqdm(range(n_trials)):
        torch.manual_seed(i)
        random.seed(i)
        scores.append(eval_full_midi(midi_path, *funcs))
    mean_score = sum(scores) / n_trials
    sd = (sum((i - mean_score) ** 2 for i in scores) / len(scores)) ** 1 / 2
    print(f"Evaluated metric is {mean_score}, sd: {sd}")
    return mean_score


if __name__ == '__main__':
    """
    Metric Evaluations
    """
    # convert_kaggle_mlm('/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/rawordinal+.pth')
    # run_mlm_test('/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/kg_rawmlm.pth')
    # run_mlm_test('/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/velocitymlm.pth')
    # run_mlm_test('/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/kg_rawordinal.pth')

    # test_eval_full_midi(
    #     '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid',
    #     '/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/kg_rawmlm.pth',
    #     mask_all=True,
    #     overlap=0.5,
    #     n_trials=10,
    #     mode='raw'
    # )
    # test_eval_full_midi(
    #     '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid',
    #     '/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/kg_rawmlm.pth',
    #     mask_all=True,
    #     overlap=0.5,
    #     n_trials=10,
    #     mode='context'
    # )

    """
    Applications / Demo
    """
    seed = 1
    midi_path = '/Users/kurono/Desktop/02714_BminorAllamande.mid'
    ckpt_path = '/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/kg_rawmlm.pth'
    mask_mode = 'some'

    torch.manual_seed(seed)
    random.seed(seed)

    # render_seq(midi_path=midi_path,
    #            ckpt_path=ckpt_path,
    #            mask_mode=mask_mode,
    #            file_stem='demo1')

    render_contextual_seq(
        midi_path=midi_path,
        ckpt_path=ckpt_path,
        mask_mode=mask_mode,
        file_stem='contextdemo1')
