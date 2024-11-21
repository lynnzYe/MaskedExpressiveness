import pretty_midi
import note_seq
import torch
from note_seq import PerformanceEvent
import numpy as np
from scipy.io.wavfile import write as write_wav
from maskexp.definitions import OUTPUT_DIR, ROOT_DIR, VELOCITY_MASK_EVENT
from note_seq.testing_lib import add_track_to_sequence


def note_sequence_to_performance_events(note_sequence, steps_per_quarter=100):
    quantized_note_sequence = note_seq.sequences_lib.quantize_note_sequence(note_sequence, steps_per_quarter)
    perfs = note_seq.performance_lib.BasePerformance._from_quantized_sequence(quantized_note_sequence, 0,
                                                                              num_velocity_bins=32,
                                                                              max_shift_steps=100)

    return perfs


def decode_output_ids(out_ids, decoder):
    """
    Decode class id into performance events
    :param out_ids:
    :param decoder:
    :return:
    """
    if isinstance(out_ids, torch.Tensor):
        return [decoder.decode_event(e.item()) for e in out_ids]
    else:
        assert isinstance(out_ids, list)
        return [decoder.decode_event(e) for e in out_ids]


def write_ids_to_midi(tokens: torch.Tensor, perf_config, output_dir=None, fstem=None):
    """
    Convert ids to MIDI, (e.g. [339, 2, 42] -> MIDI file)
    :param tokens:
    :param perf_config:
    :return:
    """
    tokenized_midi = decode_output_ids(tokens, perf_config.encoder_decoder._one_hot_encoding)
    write_token_to_midi(tokenized_midi, perf_config)


def write_token_to_midi(tokens, perf_config, output_dir=None, fstem=None):
    midi = performance_events_to_pretty_midi(tokens, perf_config.encoder_decoder._one_hot_encoding)
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if fstem is None:
        fstem = 'ids_to_mid'
    midi.write(OUTPUT_DIR + f"/{fstem}.mid")


def performance_events_to_pretty_midi(events: list[PerformanceEvent], steps_per_second=100, n_velocity_bin=32,
                                      mask_notes_without_velocity=False):
    # performance = note_seq.Performance(
    #     quantized_sequence=None,
    #     steps_per_second=steps_per_second,
    #     num_velocity_bins=n_velocity_bin)
    # for e in events:
    #     performance.append(e)
    # ns = performance.to_sequence()
    # return note_seq.midi_io.note_sequence_to_pretty_midi(ns)

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    time = 0
    active_notes = {}
    current_velocity = 1

    for event in events:
        if event.event_type == PerformanceEvent.TIME_SHIFT:
            time += event.event_value / note_seq.DEFAULT_QUARTERS_PER_MINUTE
        elif event.event_type == PerformanceEvent.NOTE_ON:
            if event.event_value == VELOCITY_MASK_EVENT.event_value:
                current_velocity = 1
                continue
            note = pretty_midi.Note(
                velocity=current_velocity,
                pitch=event.event_value,
                start=time,
                end=time
            )
            if mask_notes_without_velocity:
                current_velocity = 1
            active_notes[event.event_value] = note
            instrument.notes.append(note)
        elif event.event_type == PerformanceEvent.NOTE_OFF:
            if event.event_value in active_notes:
                active_notes[event.event_value].end = time
                del active_notes[event.event_value]
        elif event.event_type == PerformanceEvent.VELOCITY:
            current_velocity = note_seq.performance_lib.velocity_bin_to_velocity(event.event_value, n_velocity_bin)

    pm.instruments.append(instrument)
    return pm


def syn_perfevent(perf: list[PerformanceEvent], filename='rendered.wav', n_velocity_bin=32):
    return syn_prettymidi(performance_events_to_pretty_midi(perf, n_velocity_bin=n_velocity_bin),
                          filename=filename)


def syn_prettymidi(midi: pretty_midi.PrettyMIDI, output_dir=OUTPUT_DIR, filename='rendered.wav'):
    audio_data = midi.fluidsynth(sf2_path=ROOT_DIR + '/data/soundfount/grandmistral.sf2')
    audio_data = audio_data / np.max(np.abs(audio_data))
    write_wav(output_dir + '/' + filename, 44100, audio_data)
    return audio_data


def test_perf_to_midi():
    events = [
        PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT, event_value=120),
        PerformanceEvent(event_type=PerformanceEvent.NOTE_ON, event_value=60),
        PerformanceEvent(event_type=PerformanceEvent.VELOCITY, event_value=80),
        PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT, event_value=480),
        PerformanceEvent(event_type=PerformanceEvent.NOTE_OFF, event_value=60),
    ]
    midi = performance_events_to_pretty_midi(events)


def test_noteset_to_perf():
    note_sequence = note_seq.NoteSequence()
    notes = [
        # p, vel, on,  off
        (60, 60, 0.0, 0.5),  # C4
        (60, 65, 0.5, 1.0),  # C4
        (67, 66, 1.0, 1.5),  # G4
        (67, 68, 1.5, 2.0),  # G4
        (69, 72, 2.0, 2.5),  # A4
        (69, 69, 2.5, 3.0),  # A4
        (67, 67, 3.0, 4.0),  # G4
        (65, 63, 4.0, 4.5),  # F4
        (65, 61, 4.5, 5.0),  # F4
        (64, 64, 5.0, 5.5),  # E4
        (64, 62, 5.5, 6.0),  # E4
        (62, 62, 6.0, 6.5),  # D4
        (62, 57, 6.5, 7.0),  # D4
        (60, 52, 7.0, 8.0),  # C4
    ]
    add_track_to_sequence(note_sequence, 0, notes)
    perf = note_sequence_to_performance_events(note_sequence)
    midi = performance_events_to_pretty_midi(perf, n_velocity_bin=32)
    midi.write(OUTPUT_DIR + '/test_ctp.mid')
    syn_perfevent(perf, 'test_ctp.wav')


if __name__ == '__main__':
    test_noteset_to_perf()
