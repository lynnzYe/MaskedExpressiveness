import note_seq
import pretty_midi as pm
from maskexp.magenta.pipelines import performance_pipeline, note_sequence_pipelines
from maskexp.magenta.contrib import training as contrib_training
from maskexp.magenta.models.performance_rnn import performance_model
from maskexp.util.play_midi import syn_perfevent, note_sequence_to_performance_events


def extract_raw_tokens(seq_example):
    feature_lists = seq_example.feature_lists
    result = {}
    for key, feature_list in feature_lists.feature_list.items():
        # Extract values from the feature list
        if key == 'inputs':
            continue
        values = []
        for feature in feature_list.feature:
            # Assuming the features are float lists (you can adapt this to your needs)
            if feature.HasField('float_list'):
                values.append(feature.float_list.value[0])
            elif feature.HasField('int64_list'):
                values.append(feature.int64_list.value[0])
            elif feature.HasField('bytes_list'):
                values.append(feature.bytes_list.value[0])
            else:
                raise ValueError("Unsupported feature type")
        result[key] = values
    return result['labels']


def test_extract_tokens_from_midi(filepath):
    seq = note_seq.midi_io.midi_to_note_sequence(pm.PrettyMIDI(filepath))
    config = performance_model.PerformanceRnnConfig(None,
                                                    note_seq.OneHotEventSequenceEncoderDecoder(
                                                        note_seq.PerformanceOneHotEncoding()
                                                    ),
                                                    contrib_training.HParams())
    pipeline_inst = performance_pipeline.get_pipeline(
        min_events=32,
        max_events=512,
        eval_ratio=0.1,
        config=config)
    result = pipeline_inst.transform(seq)
    token_list = []
    for e in result['training_performances']:
        token_list.append(extract_raw_tokens(e))
    return token_list


def get_noteseq(filepath):
    return note_seq.midi_io.midi_to_note_sequence(pm.PrettyMIDI(filepath))


def extract_tokens_from_midi(filepath, config=None):
    """

    :param filepath:
    :param config:  performance_model.default_configs
    :return:
    """
    if config is None:
        config = performance_model.default_configs['performance']

    seq = get_noteseq(filepath=filepath)
    pipeline_inst = performance_pipeline.get_pipeline(
        min_events=32,
        max_events=512,
        eval_ratio=0.1,
        config=config)
    result = pipeline_inst.transform(seq)
    token_list = []
    for e in result['training_performances']:
        token_list.append(extract_raw_tokens(e))
    return token_list


def decode_tokens(tokens, decoder=None):
    if decoder is None:
        decoder = note_seq.OneHotEventSequenceEncoderDecoder(note_seq.PerformanceOneHotEncoding())
    out = []
    for t in tokens:
        out.append(decoder.class_index_to_event(t, None))
    return out


def get_raw_performance(filepath):
    seq = get_noteseq(filepath)
    sus = note_sequence_pipelines.SustainPipeline()
    seq = sus.transform(seq)[0]
    return note_sequence_to_performance_events(seq, steps_per_quarter=50)


if __name__ == '__main__':
    midi_path = '../../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid'
    config = performance_model.default_configs['performance_with_dynamics']
    test = extract_tokens_from_midi(
        midi_path,
        config=config
    )

    perf = decode_tokens(test[0], decoder=config.encoder_decoder)

    og_perf = get_raw_performance(midi_path)
    syn_perfevent(og_perf, filename='og.wav', n_velocity_bin=127)
    syn_perfevent(perf, filename='quantized.wav')
