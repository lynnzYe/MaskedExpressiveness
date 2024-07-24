"""
Prepare training / evaluation / test data for Bert training
"""
import note_seq
import numpy as np
import torch
from tokenize_midi import extract_tokens_from_midi
from maskexp.magenta.models.performance_rnn import performance_model

PERF_PAD = 0


def onehot(label, nclass):
    one_hot = [0.0] * nclass
    one_hot[label] = 1.0
    return one_hot


def get_attention_mask(tokens, max_seq_len=128):
    """
    Return array of ones and zeros (masks)
    :param tokens:
    :param max_seq_len:
    :return:
    """
    if len(tokens) > max_seq_len:
        raise ValueError("Token seq length exceeds max_seq_len")
    if len(tokens) == max_seq_len:
        return [1] * max_seq_len
    mask = [1] * len(tokens)
    mask.extend([0] * (max_seq_len - len(tokens)))
    return mask


def convert_onehot_to_ids(onehot):
    """
    Input single onehot vector
    :param onehot:
    :return:
    """
    assert isinstance(onehot[0], (int, float))
    return np.argmax(onehot)


def process_perf_tokenseq(token_seq, perf_config, max_seq_len=128):
    """
    Return lists of labels padded to max_seq_len
    Make a copy of the ids, and pad token seq to max seq len
    :param token_seq: dict -> { 'inputs': [onehot encodings], 'labels': [list of class idx] }
    :param perf_config:
    :param max_seq_len: default events will be padded to the end till max seq len is reached
    :return:
    """
    if perf_config is None:
        raise ValueError("Config is required to use the tokenizer!")
    pad_token = perf_config.encoder_decoder.default_event_label
    # pad_onehot = onehot(pad_token, perf_config.encoder_decoder.num_classes)

    if len(token_seq) > max_seq_len:
        raise ValueError("Token seq length exceeds max_seq_len")
    if len(token_seq) == max_seq_len:
        return [convert_onehot_to_ids(e) for e in token_seq['inputs']]

    # token_seq['labels'].append([pad_token] * (max_seq_len - len(token_seq)))
    out = [convert_onehot_to_ids(e) for e in token_seq['inputs']]
    out.extend([pad_token] * (max_seq_len - len(token_seq)))
    return out


def prepare_one_midi_data(midi_path, perf_config=None, min_seq_len=64, max_seq_len=128):
    """
    Extract sequences of input ids from midi, pad with default events after the end
    Return shape:
     - [[input ids], [...]]
     - [[attention masks], [...]]

    Token Encoding:
    1: NOTEON
    2: NOTEOFF
    3: TIMESHIFT
    4: VELOCITY
    5: (DURATION) (used only in note-based encoding)

    Decoding:
    for range [1~5]
    - if < MIN~MAX:
        - event type = curr
        - event id = get(value)
    - else:
        - offset += RANGE
        - event type ++

    :param midi_path:
    :param perf_config:
    :param max_seq_len:
    :return:
    """
    if perf_config is None:
        perf_config = performance_model.default_configs['performance']
        raise ValueError("Config is required to use the tokenizer!")

    tokens = extract_tokens_from_midi(midi_path, config=perf_config, min_seq=min_seq_len, max_seq=max_seq_len)
    attention_masks = []
    out_tokens = []
    for token_seq in tokens:
        attention_masks.append(get_attention_mask(token_seq, max_seq_len=max_seq_len))
        out_tokens.append(process_perf_tokenseq(token_seq, perf_config, max_seq_len=max_seq_len))
    return torch.tensor(out_tokens), attention_masks


def is_special_token(token: int, decoder: note_seq.encoder_decoder.OneHotEncoding, special_class_ids=None):
    """
    Determine if a token is special token
    :param token:
    :param decoder:
    :param special_class_ids:
    :return:
    """
    if special_class_ids is None:
        special_class_ids = (note_seq.PerformanceEvent.TIME_SHIFT,
                             note_seq.PerformanceEvent.VELOCITY,
                             note_seq.PerformanceEvent.DURATION)
    event = decoder.decode_event(token)
    if event.event_type in special_class_ids:
        return True
    return False


def mask_perf_tokens(token_ids: torch.tensor, perf_config=None, mask_prob=0.15):
    """
    Mask performance tokens:
    - only timeshift and velocity events will be masked
    - can be tested on different masking strategies (complete mask vs relative mask)

    :param token_ids:
    :param perf_config:
    :return:
    """
    if perf_config is None:
        perf_config = performance_model.default_configs['performance']
        raise ValueError("Config is required to use the tokenizer!")
    prob_matrix = torch.full(token_ids.shape, mask_prob)
    labels = token_ids.clone()

    # Obtain special tokens
    tokenizer = perf_config.encoder_decoder._one_hot_encoding
    special_token_mask = [is_special_token(val, tokenizer,
                                           special_class_ids=(note_seq.PerformanceEvent.TIME_SHIFT,
                                                              note_seq.PerformanceEvent.VELOCITY))
                          for val in token_ids.tolist()]
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100


def prepare_data(midi_file_list, train=0.8, val=0.1, test=0.1):
    data = []
    for file in midi_file_list:
        tokens, masks = prepare_one_midi_data(file)


def main():
    midi_path = '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid'
    perf_config = performance_model.default_configs['performance']
    tokens, mask = prepare_one_midi_data(midi_path, perf_config=perf_config)

    mask_perf_tokens(tokens, perf_config)


if __name__ == '__main__':
    main()
