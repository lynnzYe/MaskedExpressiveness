"""
@author: Bmois
@brief: using match file and score info, fill in the missing midi notes
"""
from logging import warning
from nis import match

from bokeh.sampledata.sprint import sprint
from scipy.stats import alpha
from tensorflow.python.framework.errors_impl import OutOfRangeError

from maskexp.util.alignment_parser import MatchFileParser, ScoreParser, SprParser
import pretty_midi
from dataclasses import dataclass

ONSET_DEVIANCE = 0.001
NOTE_TO_OFFSET = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11
}
ma_alpha = 0.01


def pitchname_to_midi(name):
    note = name[:-1]
    octave = int(name[-1])
    midi_number = 12 * (octave + 1) + NOTE_TO_OFFSET[note]
    return midi_number


def _infer_time(target_y, onset1, stime1, onset2, stime2):
    # Linear curve fitting on current two data points
    # Calculate onset time = a * score_time + b
    a_st = (onset1 - onset2) / (stime1 - stime2)
    b_st = onset1 - stime1 * a_st

    onset = a_st * target_y + b_st

    offset = onset2  # For simplicity, just end right at where the next note starts
    if onset > onset1 and onset > onset2:
        offset = onset + 2  # Case when offset is hard to be properly inferred, extend 2s
    assert offset > onset
    return onset, offset


def infer_aligned_midi_time(score_onset_time, score_offset_time, match_info: MatchFileParser, spr_info: SprParser):
    """
    Return the onset and offset time of a given note,

    :param score_onset_time: the onset time recorded in match file (not spr!) (They are different!)
    :param score_offset_time: similar to above
    :param match_info:
    :param spr_info:
    :param moving_avg: for tempo smoothing
    :return:
    """
    # First, we need to find the closest performed onset to the score MIDI note
    assert len(spr_info.sorted_notes) == len(match_info.sorted_match_notes)
    idx_search = 0
    for idx in reversed(range(len(match_info.sorted_match_notes))):
        note = match_info.sorted_match_notes[idx]
        if score_onset_time > note['onset_time'] + ONSET_DEVIANCE:
            break
        idx_search = idx

    # Then we need to infer the onset time
    # - Assume that the best way to infer current tempo is by referring to the adjacent two notes (instead of distant)
    total_note_count = len(match_info.sorted_match_notes)

    idx1 = idx_search - 1 if idx_search < total_note_count else total_note_count - 2
    idx2 = idx_search if idx_search < total_note_count else total_note_count - 1
    if idx1 < 0:
        idx1 += 1
        idx2 += 1

    # Linear Interpolate: real_onset_time = a * score_onset_time + b
    real_onset1 = spr_info.sorted_notes[idx1]['onset_time']
    real_onset2 = spr_info.sorted_notes[idx2]['onset_time']
    a_os, b_os = linear_interpolate(real_onset1, real_onset2,
                                    match_info.sorted_match_notes[idx1]['onset_time'],
                                    match_info.sorted_match_notes[idx2]['onset_time'])
    # print(a_os, b_os, idx1, idx2)
    return score_onset_time * a_os + b_os, score_offset_time * a_os + b_os


def infer_aligned_note_time(note_stime, match_info: MatchFileParser):
    """
    Return the onset and offset time of a given note stime,
    if it can be matched to a performed note, else infer according to adjacent notes

    :param note_stime: the score time of note to be searched
    :param match_info:
    :return:
    """
    # Case 1: the note is in a chord containing a matched note (assume simultaneous onset)
    idx_search = 0
    for idx, note in enumerate(match_info.sorted_match_notes):
        if note['score_time'] > note_stime:
            break
        elif note['score_time'] == note_stime:
            return note['onset_time'], note['offset_time']
        idx_search += 1

    # Case 2: We need to infer the onset time
    # - Assume that the best way to infer current tempo is by referring to the adjacent two notes (instead of distant)
    total_note_count = len(match_info.sorted_match_notes)
    note1 = match_info.sorted_match_notes[idx_search - 1] \
        if idx_search < total_note_count else match_info.sorted_match_notes[total_note_count - 2]
    note2 = match_info.sorted_match_notes[idx_search] \
        if idx_search < total_note_count else match_info.sorted_match_notes[total_note_count - 1]
    onset1 = note1['onset_time']
    onset2 = note2['onset_time']
    stime1 = note1['score_time']
    stime2 = note2['score_time']

    return _infer_time(note_stime, onset1, stime1, onset2, stime2)


def create_midi(pitch, onset, offset, velocity):
    assert offset > onset
    return pretty_midi.Note(pitch=pitch, start=onset, end=offset, velocity=velocity)


@dataclass
class MIDINote:
    pitch: int
    start: float
    end: float
    velocity: int

    def __post_init__(self):
        assert self.end > self.start
        assert 0 <= self.velocity < 128


def restrict_midi(midi_meta_list):
    # sort midi meta list by onset first
    midi_meta_list = sorted(midi_meta_list, key=lambda mid: mid.start)
    clean_midi_list = []
    delete_indices = []
    for idx in reversed(range(len(midi_meta_list) - 1)):
        if idx == 0:
            break
        idx_search = idx
        if idx in delete_indices or midi_meta_list[idx].start == midi_meta_list[idx].end:
            continue
        while idx_search > 0:
            idx_search -= 1
            if midi_meta_list[idx_search].pitch == midi_meta_list[idx].pitch:
                if midi_meta_list[idx_search].end >= midi_meta_list[idx].start:
                    if abs(midi_meta_list[idx_search].start - midi_meta_list[idx].start) <= ONSET_DEVIANCE:
                        # print("\x1B[33m[Warning]\033[0m Two simultaneous note-on with the same pitch! Delete one")
                        # print('Pitch:', midi_meta_list[idx])
                        delete_indices.append(idx_search if midi_meta_list[idx_search].end < midi_meta_list[idx].end \
                                                  else idx)
                        continue
                    # Overlapped note found. Trim previous note.
                    midi_meta_list[idx_search].end = max(midi_meta_list[idx].start - 0.01,
                                                         midi_meta_list[idx_search].start + 0.0001)
                    assert midi_meta_list[idx_search].start < midi_meta_list[idx_search].end
                    # minus 10ms to prevent normalization error
                    break
        if idx not in delete_indices:
            clean_midi_list.append(midi_meta_list[idx])
    return clean_midi_list


def get_spr_midi_time(spr_info: SprParser, idx: str):
    return spr_info.notes_by_id[idx]['onset_time'], spr_info.notes_by_id[idx]['offset_time']


def midify_score_align(score_info: ScoreParser, match_info: MatchFileParser):
    """
    There are two types of missing notes:
    - onset-aware: within a chord matched to a performed note(s)
    - onset-agnostic: no matching performed note. Have to infer onset time
    :param match_info:
    :param score_info:
    :return:
    """
    med_midi_meta_list = []
    midi_meta_list = []
    # Fill in performed MIDI events
    for note in match_info.matched_notes.values():
        if note['offset_time'] == note['onset_time']:
            continue  # Skip inaudible notes
        assert note['offset_time'] > note['onset_time']

        med_midi_meta_list.append(
            MIDINote(pitchname_to_midi(note['pitch']),
                     note['onset_time'],
                     note['offset_time'],
                     note['onset_velocity'])
        )
    restrict_midi(med_midi_meta_list)

    for note in match_info.missing_notes:
        missing_id = note[0]
        st = score_info.get_attr_by_id(missing_id, 'score_time')
        onset, offset = infer_aligned_note_time(st, match_info)
        pt = pitchname_to_midi(score_info.get_attr_by_id(missing_id, 'pitch'))
        midi_meta_list.append(MIDINote(pitch=pt, start=onset, end=offset, velocity=1))
    restrict_midi(midi_meta_list)

    med_midi_list = []
    midi_list = []
    for mid in med_midi_meta_list:
        med_midi_list.append(create_midi(mid.pitch, mid.start, mid.end, mid.velocity))
    for i, mid in enumerate(midi_meta_list):
        midi_list.append(create_midi(mid.pitch, mid.start, mid.end, mid.velocity))
    return med_midi_list, midi_list


def linear_interpolate(y1, y2, x1, x2):
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    return a, b


def find_2_mapped_midi(spr_info: SprParser, match_info: MatchFileParser):
    """
    Spr info records performed MIDI events. Match info contains MIDI alignment info.
    A 100% successful alignment means a one-to-one mapping from spr_info to match_info.matched_notes

    :param spr_info:
    :param match_info:
    :param n_search:
    :return:
    """
    ref_idx = 0
    tgt_idx = ref_idx + 1
    if len(spr_info.notes_by_id) == len(match_info.sorted_match_notes):
        # Return the first two mapped MIDI with different onset time
        while spr_info.sorted_notes[tgt_idx]['onset_time'] == match_info.sorted_match_notes[ref_idx]['onset_time']:
            tgt_idx += 1
        assert spr_info.sorted_notes[ref_idx]['pitch'] == match_info.sorted_match_notes[ref_idx]['pitch']
        assert spr_info.sorted_notes[tgt_idx]['pitch'] == match_info.sorted_match_notes[tgt_idx]['pitch']
        return [ref_idx, tgt_idx], [match_info.sorted_match_notes[ref_idx], match_info.sorted_match_notes[tgt_idx]]

    # Else, it is not a one-to-one mapping
    sref_idx, stgt_idx = ref_idx, tgt_idx
    while spr_info.sorted_notes[sref_idx]['pitch'] != match_info.sorted_match_notes[ref_idx]['pitch']:
        # TODO @Bmois: make this search mechanism more robust and safe
        sref_idx += 1
        if sref_idx >= len(spr_info.sorted_notes):
            warning("Cannot find a mapped midi between spr and match")
            return None, None

    while spr_info.notes_by_id[stgt_idx] != match_info.sorted_match_notes[tgt_idx]['pitch']:
        stgt_idx += 1
        if stgt_idx >= len(spr_info.sorted_notes):
            warning("Cannot find a secomd mapped midi between spr and match")
            return None, None
    return [ref_idx, tgt_idx], [match_info.sorted_match_notes[ref_idx], match_info.sorted_match_notes[tgt_idx]]


def midify_midi_align(spr_info: SprParser, match_info: MatchFileParser):
    """

    :param spr_info:
    :param match_info:
    :return:
    """
    med_midi_meta_list = []
    midi_meta_list = []
    assert len(spr_info.notes_by_id) >= len(match_info.matched_notes)

    # Each Spr id map to exactly one score id, in order
    # First, loop the performed MIDI so that all of them are in the output MIDI
    for note in spr_info.notes_by_id.values():
        if note['offset_time'] == note['onset_time']:
            continue  # Skip inaudible notes
        assert note['offset_time'] > note['onset_time']
        med_midi_meta_list.append(
            MIDINote(pitchname_to_midi(note['pitch']),
                     note['onset_time'],
                     note['offset_time'],
                     note['onset_velocity'])
        )

    # Then, loop all the extra notes (i.e. the rest of the notes in the score)
    if len(match_info.extra_notes) == 0:
        print("\x1B[33m[Warning]\033[0m MIDI alignment found no score notes to be added")

    for note in match_info.extra_notes:
        # because we align score MIDI to performed MIDI, extra notes are actually missing score notes.
        onset, offset = infer_aligned_midi_time(note['onset_time'], note['offset_time'], match_info, spr_info)
        if onset == offset:
            continue  # Skip inaudible notes
        midi_meta_list.append(MIDINote(pitch=pitchname_to_midi(note['pitch']), start=onset, end=offset, velocity=1))
    restrict_midi(midi_meta_list)

    med_midi_list = []
    midi_list = []
    for mid in med_midi_meta_list:
        med_midi_list.append(create_midi(mid.pitch, mid.start, mid.end, mid.velocity))
    for i, mid in enumerate(midi_meta_list):
        midi_list.append(create_midi(mid.pitch, mid.start, mid.end, mid.velocity))
    return med_midi_list, midi_list


def find_missing_midi(score_info: ScoreParser, match_info: MatchFileParser, spr_info: SprParser or None):
    """
    There are two types of missing notes:
    - onset-aware: within a chord matched to a performed note(s)
    - onset-agnostic: no matching performed note. Have to infer onset time
    :param match_info:  for score_alignment, just pass in the match file generated
                        for midi_alignment, Score match file is anticipated!
    :param score_info:  This is the fmt3x file generated from the score's MIDI
                        (not xml because we are doing MIDI alignment)
    :param spr_info:    [Notice] anticipate spr of the performed MIDI, not the score's !!!!!
    :return:
    """
    assert score_info is not None
    if spr_info is not None:
        return midify_midi_align(spr_info, match_info)
    else:
        return midify_score_align(score_info, match_info)


def write_coop_midi(melody_midi_list, midi_list, fpath):
    """
    Input user performed midi list and score midi list, write them as two separate midi tracks
    :param melody_midi_list:
    :param midi_list:
    :param fpath:
    :return:
    """
    midi = pretty_midi.PrettyMIDI()

    med_inst = pretty_midi.Instrument(program=0)
    score_inst = pretty_midi.Instrument(program=0)

    for note in melody_midi_list:
        med_inst.notes.append(note)
    for note in midi_list:
        score_inst.notes.append(note)

    midi.instruments.append(med_inst)
    midi.instruments.append(score_inst)

    midi.write(fpath)


def ftest_score_align():
    mat = MatchFileParser()
    mat.parse_file('/Users/kurono/Documents/python/GEC/ExpressiveMLM/maskexp/demo/'
                   'AlignmentTool/data/Chopin_match.txt')

    scr = ScoreParser()
    scr.parse_file('/Users/kurono/Documents/python/GEC/ExpressiveMLM/maskexp/demo/'
                   'AlignmentTool/data/ChopinPrelude-Rachmaninoff-Piano_fmt3x.txt')

    med_midi_list, midi_list = find_missing_midi(scr, mat, None)
    write_coop_midi(med_midi_list, midi_list, 'test.mid')


def ftest_midi_align():
    mat = MatchFileParser()
    mat.parse_file('/Users/kurono/Desktop/schu_score_match.txt')

    scr = ScoreParser()
    scr.parse_file('/Users/kurono/Desktop/schu_fmt3x.txt')

    spr = SprParser()
    spr.parse_file('/Users/kurono/Desktop/schu_spr.txt')

    med_midi_list, midi_list = find_missing_midi(scr, mat, spr)
    write_coop_midi(med_midi_list, midi_list, 'test.mid')


if __name__ == '__main__':
    # ftest_score_align()
    ftest_midi_align()
