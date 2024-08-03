import fluidsynth
import pretty_midi
import note_seq
from note_seq import PerformanceEvent
import numpy as np
from scipy.io.wavfile import write as write_wav
from maskexp.definitions import OUTPUT_DIR, ROOT_DIR
from note_seq.testing_lib import add_track_to_sequence


def note_sequence_to_performance_events(note_sequence, steps_per_quarter=100):
    quantized_note_sequence = note_seq.sequences_lib.quantize_note_sequence(note_sequence, steps_per_quarter)
    perfs = note_seq.performance_lib.BasePerformance._from_quantized_sequence(quantized_note_sequence, 0,
                                                                              num_velocity_bins=127,
                                                                              max_shift_steps=10000)

    return perfs


def performance_events_to_pretty_midi(events: list[PerformanceEvent], n_velocity_bin=32):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    time = 0
    active_notes = {}
    current_velocity = 100

    for event in events:
        if event.event_type == PerformanceEvent.TIME_SHIFT:
            time += event.event_value / note_seq.DEFAULT_QUARTERS_PER_MINUTE
        elif event.event_type == PerformanceEvent.NOTE_ON:
            note = pretty_midi.Note(
                velocity=current_velocity,
                pitch=event.event_value,
                start=time,
                end=time
            )
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
    syn_perfevent(perf, 'test_ctp.wav')


if __name__ == '__main__':
    test_noteset_to_perf()
