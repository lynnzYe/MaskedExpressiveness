"""
Prepare training / evaluation / test data for Bert training
"""
import random
import note_seq
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tokenize_midi import extract_mlm_tokens_from_midi, onehot, decode_tokens
from maskexp.magenta.models.performance_rnn import performance_model

PERF_PAD = 0


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

    tokens = extract_mlm_tokens_from_midi(midi_path, config=perf_config, min_seq=min_seq_len, max_seq=max_seq_len)
    assert len(tokens) > 0
    attention_masks = []
    out_tokens = []
    for token_dict in tokens:
        attention_masks.append(get_attention_mask(token_dict['inputs'], max_seq_len=max_seq_len))
        out_tokens.append(process_perf_tokenseq(token_dict['inputs'], perf_config, max_seq_len=max_seq_len))
    return torch.tensor(out_tokens), torch.tensor(attention_masks)


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


def find_event_range(decoder, class_idx):
    assert isinstance(decoder, note_seq.EventSequenceEncoderDecoder)
    for e in decoder._one_hot_encoding._event_ranges:
        if e[0] == class_idx:
            return (note_seq.PerformanceEvent(event_type=class_idx,
                                              event_value=e[1]),
                    note_seq.PerformanceEvent(event_type=class_idx,
                                              event_value=e[2]))
    raise ValueError("Cannot find class", class_idx, 'event range')


def random_token_walk(token_id, decoder, random_range=-1):
    event = decoder._one_hot_encoding.decode_event(token_id)
    min_e, max_e = find_event_range(decoder, event.event_type)
    if random_range == -1:
        random_range = max_e.event_value
    assert min_e.event_value <= random_range <= max_e.event_value

    min_class_id = decoder._one_hot_encoding.encode_event(min_e)
    max_class_id = decoder._one_hot_encoding.encode_event(max_e)

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
    special_token_mask = [is_special_token(val, tokenizer, special_class_ids=special_ids)
                          for val in token_ids.tolist()]
    prob_matrix.masked_fill_(torch.tensor(special_token_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100

    # TODO: by rule - only timeshift random number etc., or roughly scaled (beat info preserved)
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    token_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # Random work 10%
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    token_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return token_ids, labels


def prepare_data(midi_file_list, train=0.8, val=0.1, test=0.1, batch_size=32, perf_config=None):
    if perf_config is None:
        raise ValueError("Performance config must be provided")
    token_ids = []
    masks = []
    for file in midi_file_list:
        tk, msk = prepare_one_midi_data(file, perf_config=perf_config)
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

    train_dataset = TensorDataset(train_ids, train_msks, train_labels)
    val_dataset = TensorDataset(val_ids, val_msks, val_labels)
    test_dataset = TensorDataset(test_ids, test_msks, test_labels)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


def test_prepare_data():
    midi_path = '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid'
    perf_config = performance_model.default_configs['performance']
    prepare_data([midi_path], perf_config=perf_config)


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
    # test_prepare_data()
    # test_mask_perf()
    test_random_token_walk()


if __name__ == '__main__':
    main()
