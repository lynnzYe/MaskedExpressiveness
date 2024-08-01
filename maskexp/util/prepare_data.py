"""
Prepare training / evaluation / test data for Bert training
"""
import random
import note_seq
import tqdm
import numpy as np
import torch
from maskexp.util.tokenize_midi import extract_tokens_from_midi, onehot, decode_tokens
from maskexp.magenta.models.performance_rnn import performance_model
from maskexp.definitions import DATA_DIR
from pathlib import Path
from maskexp.definitions import IGNORE_LABEL_INDEX, DEFAULT_MASK_EVENT


def get_attention_mask(tokens, max_seq_len):
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


def pad_onehot_perf_tokens(token_seq, perf_config, max_seq_len=128):
    """
    Return lists of "labels" padded to max_seq_len
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
    assert isinstance(token_seq, (list, torch.tensor))
    if len(token_seq) > max_seq_len:
        raise ValueError("Token seq length exceeds max_seq_len")
    if len(token_seq) == max_seq_len:
        return [convert_onehot_to_ids(e) for e in token_seq]

    # token_seq['labels'].append([pad_token] * (max_seq_len - len(token_seq)))
    out = [convert_onehot_to_ids(e) for e in token_seq]
    out.extend([pad_token] * (max_seq_len - len(out)))
    assert len(out) == max_seq_len
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
    assert len(tokens) > 0
    attention_masks = []
    out_tokens = []
    for token_dict in tokens:
        attention_masks.append(get_attention_mask(token_dict['inputs'], max_seq_len=max_seq_len))
        out_tokens.append(pad_onehot_perf_tokens(token_dict['inputs'], perf_config, max_seq_len=max_seq_len))
    return torch.tensor(out_tokens), torch.tensor(attention_masks)


def only_mask_special_token(tokens: list, decoder: note_seq.encoder_decoder.OneHotEncoding, special_class_ids=None):
    """
    Determine if a token is special token
    :param tokens:
    :param decoder:
    :param special_class_ids:
    :return:
    """
    if special_class_ids is None:
        special_class_ids = (note_seq.PerformanceEvent.TIME_SHIFT,
                             note_seq.PerformanceEvent.VELOCITY,
                             note_seq.PerformanceEvent.DURATION)
    return [int(decoder.decode_event(e).event_type not in special_class_ids) for e in tokens]


def find_event_range(decoder, class_idx):
    assert isinstance(decoder, note_seq.encoder_decoder.OneHotEncoding)
    for e in decoder._event_ranges:
        if e[0] == class_idx:
            return (note_seq.PerformanceEvent(event_type=class_idx,
                                              event_value=e[1]),
                    note_seq.PerformanceEvent(event_type=class_idx,
                                              event_value=e[2]))
    raise ValueError("Cannot find class", class_idx, 'event range')


def random_token_walk(token_id, decoder, random_range=-1):
    event = decoder.decode_event(token_id)
    min_e, max_e = find_event_range(decoder, event.event_type)
    if random_range == -1:
        random_range = max_e.event_value
    assert min_e.event_value <= random_range <= max_e.event_value

    min_class_id = decoder.encode_event(min_e)
    max_class_id = decoder.encode_event(max_e)

    walk = random.uniform(-1 * min(random_range, token_id - min_class_id), min(random_range, max_class_id - token_id))
    result = int(token_id + walk)
    assert min_class_id <= result <= max_class_id
    return result


def mask_perf_tokens(token_ids: torch.tensor, perf_config=None, mask_prob=0.15, special_ids=None):
    """
    Mask performance tokens:
    - only timeshift and velocity events will be masked
    - can be tested on different masking strategies (complete mask vs relative mask)

    :param token_ids:
    :param perf_config:
    :param mask_prob:
    :return:
    """
    if perf_config is None:
        perf_config = performance_model.default_configs['performance']
        raise ValueError("Config is required to use the tokenizer!")
    if special_ids is None:
        special_ids = (note_seq.PerformanceEvent.VELOCITY,
                       # note_seq.PerformanceEvent.TIME_SHIFT
                       # TODO: Time shift needs extra logic
                       )

    prob_matrix = torch.full(token_ids.shape, mask_prob)
    labels = token_ids.clone()

    # Obtain special tokens
    tokenizer = perf_config.encoder_decoder._one_hot_encoding
    special_token_mask = [only_mask_special_token(val, tokenizer, special_class_ids=special_ids)
                          for val in token_ids.tolist()]
    prob_matrix.masked_fill_(torch.tensor(special_token_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = IGNORE_LABEL_INDEX

    # TODO: Implement mechanism for timeshift
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # default mask token is an unused midi
    token_ids[indices_replaced] = tokenizer.encode_event(DEFAULT_MASK_EVENT)

    # Random work 10%
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    indices_to_replace = torch.nonzero(indices_random, as_tuple=True)

    for i in range(len(indices_to_replace[0])):
        batch_idx = indices_to_replace[0][i].item()
        seq_idx = indices_to_replace[1][i].item()
        token_ids[batch_idx, seq_idx] = random_token_walk(token_ids[batch_idx, seq_idx].item(), tokenizer)

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return token_ids, labels


def prepare_token_data(midi_file_list, train=0.8, val=0.1, perf_config=None, min_seq_len=64, max_seq_len=128):
    if perf_config is None:
        raise ValueError("Performance config must be provided")
    token_ids = []
    masks = []
    for file in tqdm.tqdm(midi_file_list):
        tk, msk = prepare_one_midi_data(file, perf_config=perf_config, min_seq_len=min_seq_len, max_seq_len=max_seq_len)
        token_ids.extend(tk)
        masks.extend(msk)
    token_ids = torch.stack(token_ids, dim=0)
    masks = torch.stack(masks, dim=0)

    # Split to train val test
    total_size = len(token_ids)
    train_size = int(total_size * train)
    val_size = int(total_size * val)
    test_size = total_size - train_size - val_size

    train_ids, val_ids, test_ids = torch.split(token_ids, [train_size, val_size, test_size])
    train_msks, val_msks, test_msks = torch.split(masks, [train_size, val_size, test_size])

    train_labels = torch.zeros(train_size)
    val_labels = torch.zeros(val_size)
    test_labels = torch.zeros(test_size)

    return {
        'train': {'ids': train_ids, 'msks': train_msks, 'labels': train_labels},
        'val': {'ids': val_ids, 'msks': val_msks, 'labels': val_labels},
        'test': {'ids': test_ids, 'msks': test_msks, 'labels': test_labels}
    }


def save_datadict(data, save_dir=DATA_DIR, save_name='dataset'):
    path = Path(save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)
    torch.save(data, f'{save_dir}/{save_name}.pt')


def test_prepare_token_data():
    midi_path = '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid'
    perf_config = performance_model.default_configs['performance']
    result = prepare_token_data([midi_path], perf_config=perf_config)
    save_datadict(result)
    pass


def test_mask_perf():
    midi_path = '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid'
    perf_config = performance_model.default_configs['performance_with_dynamics']
    tokens, mask = prepare_one_midi_data(midi_path, perf_config=perf_config)

    mask_perf_tokens(tokens[0, :], perf_config)


def test_random_token_walk():
    perf_config = performance_model.default_configs['performance_with_dynamics']
    event = note_seq.PerformanceEvent(event_type=4, event_value=10)
    tid = perf_config.encoder_decoder._one_hot_encoding.encode_event(event)
    for i in range(100):
        token = random_token_walk(token_id=tid, decoder=perf_config.encoder_decoder)


def main():
    test_prepare_token_data()
    # test_mask_perf()
    # test_random_token_walk()


if __name__ == '__main__':
    main()
