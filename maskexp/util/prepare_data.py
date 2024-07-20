"""
Prepare training / evaluation / test data for Bert training
"""
from tokenize_midi import extract_tokens_from_midi
from maskexp.magenta.models.performance_rnn import performance_model

PERF_PAD = 0


def prepare_one_midi_data(midi_path, perf_config=None, max_seq_len=128):
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

    # perf_config.encoder_decoder.class_index_to_event(PERF_PAD, None)
    tokens = extract_tokens_from_midi(midi_path, config=perf_config)
    for t in tokens:
        tokens = t['labels']




if __name__ == '__main__':
    midi_path = '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid'
    prepare_one_midi_data(midi_path)
