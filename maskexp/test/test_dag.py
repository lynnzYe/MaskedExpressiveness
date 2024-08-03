import note_seq
from note_seq import testing_lib
from maskexp.magenta.pipelines import performance_pipeline
from maskexp.magenta.models.performance_rnn import performance_model
from maskexp.magenta.contrib import training as contrib_training


def test_perfextract():
    note_sequence = note_seq.NoteSequence()
    config = performance_model.PerformanceRnnConfig(
        None,
        note_seq.OneHotEventSequenceEncoderDecoder(
            note_seq.PerformanceOneHotEncoding()), contrib_training.HParams())
    testing_lib.add_track_to_sequence(note_sequence, 0,
                                      [(36, 100, 0.00, 2.0),
                                       (40, 55, 2.1, 5.0),
                                       (44, 80, 3.6, 5.0),
                                       (41, 45, 5.1, 8.0),
                                       (64, 100, 6.6, 10.0),
                                       (55, 120, 8.1, 11.0),
                                       (39, 110, 9.6, 9.7),
                                       (53, 99, 11.1, 14.1),
                                       (51, 40, 12.6, 13.0),
                                       (55, 100, 14.1, 15.0),
                                       (54, 90, 15.6, 17.0),
                                       (60, 100, 17.1, 18.0)])

    pipeline_inst = performance_pipeline.get_pipeline(
        min_events=32,
        max_events=512,
        eval_ratio=0,
        config=config)
    result = pipeline_inst.transform(note_sequence)
    return result


if __name__ == '__main__':
    test_perfextract()
