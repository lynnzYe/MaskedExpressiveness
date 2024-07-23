"""
Prepare training / evaluation / test data for Bert training
"""
import note_seq

from tokenize_midi import extract_tokens_from_midi
from maskexp.magenta.models.performance_rnn import performance_model

PERF_PAD = 0


def onehot(label, nclass):
    one_hot = [0.0] * nclass
    one_hot[label] = 1.0
    return one_hot


def pad_tokenseq(tokens, perf_config, max_seq_len=128):
    """

    :param tokens: dict -> { 'inputs': [onehot encodings], 'labels: [list of class idx] }
    :param perf_config:
    :param max_seq_len: default events will be padded to the end till max seq len is reached
    :return:
    """
    if perf_config is None:
        raise ValueError("Config is required to use the tokenizer!")
    pad_token = perf_config.encoder_decoder.default_event_label
    pad_onehot = onehot(pad_token, perf_config.encoder_decoder.num_classes)

    if len(tokens) > max_seq_len:
        raise ValueError("Token seq length exceeds max_seq_len")
    if len(tokens) == max_seq_len:
        return tokens
    tokens['labels'].append([pad_token] * (max_seq_len - len(tokens)))
    tokens['inputs'].append([pad_onehot] * (max_seq_len - len(tokens)))
    return tokens


def prepare_one_midi_data(midi_path, perf_config=None, min_seq_len=64, max_seq_len=128):
    """
    Extract tokens, pad with MIDI=0 to the back

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

    for token_seq in tokens:
        pad_tokenseq(token_seq, perf_config, max_seq_len=max_seq_len)


if __name__ == '__main__':
    midi_path = '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid'
    perf_config = performance_model.default_configs['performance']
    prepare_one_midi_data(midi_path, perf_config=perf_config)
