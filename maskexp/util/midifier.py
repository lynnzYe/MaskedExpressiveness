"""
@author: Bmois
@brief: using match file and score info, fill in the missing midi notes
"""

from parser import MatchFileParser, ScoreParser
import pretty_midi
from dataclasses import dataclass

NOTE_TO_OFFSET = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}


def pitchname_to_midi(name):
    note = name[:-1]
    octave = int(name[-1])
    midi_number = 12 * (octave + 1) + NOTE_TO_OFFSET[note]
    return midi_number


def get_aligned_note_time(note_id, match_info: MatchFileParser, score_info: ScoreParser):
    """
    Return the onset and offset time of a given note id,
    if it can be matched to a performed note, else infer according to adjacent notes

    :param note_id:
    :param match_info:
    :param score_info:
    :return:
    """
    assert note_id not in match_info.notes_by_id.keys()  # because it should be a missing note
    # Case 1: the note is in a chord containing a matched note
    idx_search = 0
    note_stime = score_info.notes_by_id[note_id]['score_time']
    for idx, note in enumerate(match_info.sorted_notes):
        idx_search = idx
        if note[1] > note_stime:
            break
        elif note[1] == note_stime:
            return (match_info.get_attr_by_id(note[0], 'onset_time'),
                    match_info.get_attr_by_id(note[0], 'offset_time'))

    # Case 2: We need to infer the onset time
    prev_note = match_info.sorted_notes[idx_search - 1] if idx_search > 0 else None
    next_note = match_info.sorted_notes[idx_search]

    def infer_time(target_stime, prev_stime, next_stime, prev_onset, next_onset):
        assert next_onset >= prev_onset and next_stime >= prev_stime
        ratio = (target_stime - prev_stime) / (next_stime - prev_stime)

        onset = prev_onset + ratio * (next_onset - prev_onset)
        offset = next_onset  # For simplicity, just end right at where the next note starts
        assert offset >= onset
        return onset, offset

    return infer_time(note_stime, prev_note[1], next_note[1],
                      match_info.get_attr_by_id(prev_note[0], 'onset_time'),
                      match_info.get_attr_by_id(next_note[0], 'onset_time'))


def create_midi(pitch, onset, offset, velocity):
    assert offset > onset
    return pretty_midi.Note(pitch=pitch, start=onset, end=offset, velocity=velocity)


@dataclass
class MIDIMeta:
    pitch: int
    onset: float
    offset: float
    velocity: int

    def __post_init__(self):
        assert self.offset > self.onset
        assert 0 <= self.velocity < 128


def restrict_midi(midi_meta_list):
    # sort midi meta list by onset first
    midi_meta_list = sorted(midi_meta_list, key=lambda mid: mid.onset)
    for idx in reversed(range(len(midi_meta_list))):
        if idx == 0:
            break
        idx_search = idx
        while idx_search > 0:
            idx_search -= 1
            if midi_meta_list[idx_search].pitch == midi_meta_list[idx].pitch:
                if midi_meta_list[idx_search].offset >= midi_meta_list[idx].onset:
                    midi_meta_list[idx_search].offset = midi_meta_list[idx].onset - 0.01
                    assert midi_meta_list[idx_search].onset < midi_meta_list[idx_search].offset
                    # minus 10ms to prevent normalization error
                    break


def find_missing_midi(match_info: MatchFileParser, score_info: ScoreParser):
    """
    There are two types of missing notes:
    - onset-aware: within a chord matched to a performed note(s)
    - onset-agnostic: no matching performed note. Have to infer onset time
    :param match_info:
    :param score_info:
    :return:
    """
    midi_meta_list = []
    # Fill in performed MIDI events
    for note in match_info.notes_by_id.values():
        assert note['offset_time'] > note['onset_time']
        midi_meta_list.append(
            MIDIMeta(pitchname_to_midi(note['pitch']),
                     note['onset_time'],
                     note['offset_time'],
                     note['onset_velocity'])
        )

    for note in match_info.missing_notes:
        missing_id = note[0]
        pt = pitchname_to_midi(score_info.get_attr_by_id(missing_id, 'pitch'))
        onset, offset = get_aligned_note_time(missing_id, match_info, score_info)
        midi_meta_list.append(MIDIMeta(pitch=pt, onset=onset, offset=offset, velocity=1))

    restrict_midi(midi_meta_list)

    midi_list = []
    for i, mid in enumerate(midi_meta_list):
        midi_list.append(create_midi(mid.pitch, mid.onset, mid.offset, mid.velocity))
    return midi_list


def write_midi(midi_list, fpath):
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    for note in midi_list:
        inst.notes.append(note)

    midi.instruments.append(inst)
    midi.write(fpath)


if __name__ == '__main__':
    mat = MatchFileParser()
    mat.parse_file('match_data.txt')

    scr = ScoreParser()
    scr.parse_file('score_data.txt')

    midi_list = find_missing_midi(mat, scr)
    write_midi(midi_list, 'test.mid')
