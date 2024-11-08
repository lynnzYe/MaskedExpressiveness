"""
@author: Bmois
@brief: Parser for hmm and match txt files.

fmt3x is very similar to hmm except that notes with simultaneous onsets are combined into note groups.
"""
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


class ScoreParser:
    def __init__(self):
        self.notes_by_id = {}
        self.tqpn = 0
        self.version = ''
        self.sorted_notes = []

    def parse_line(self, line):
        parts = line.strip().split("\t")
        line = line.replace('\n', '')
        if 'TPQN' in line:
            self.tqpn = int(line.strip(': ')[-1])
            return
        elif 'Fmt3xVersion' in line:
            self.version = line.strip(': ')[-1]
            return
        elif '//' in line:
            print("\x1B[34m[Info]\033[0m ", line)
            return

        # Parse the basic attributes
        score_time = float(parts[0])  # Score beat
        bar = int(parts[1])  # Bar number
        staff = int(parts[2])  # Staff number
        voice = int(parts[3])  # Voice number
        sub_voice = int(parts[4])  # Sub-voice number
        order = int(parts[5])  # Order
        event_type = parts[6]  # Event type
        duration = float(parts[7])  # Duration
        num_notes = int(parts[8])  # Number of notes in the chord

        # Parse the pitches, note types, and note IDs
        pitches = parts[9:9 + num_notes]
        note_types = parts[9 + num_notes:9 + 2 * num_notes]
        note_ids = parts[9 + 2 * num_notes:9 + 3 * num_notes]

        # Store each note by its note ID for easy access
        for i in range(num_notes):
            self.notes_by_id[note_ids[i]] = {
                'score_time': score_time,
                'bar': bar,
                'staff': staff,
                'voice': voice,
                'sub_voice': sub_voice,
                'order': order,
                'event_type': event_type,
                'duration': duration,
                'pitch': pitches[i],
                'note_type': note_types[i],
                'chord': note_ids  # Store all note IDs in the chord
            }
        self.sorted_notes.append((note_ids, score_time))

    def parse_file(self, filepath):
        with open(filepath, "r") as file:
            for line in file:
                self.parse_line(line)

    def get_note_by_id(self, note_id):
        """Retrieve a note by its ID."""
        return self.notes_by_id.get(note_id, None)

    def get_attr_by_id(self, note_id, attr):
        if note_id not in self.notes_by_id.keys():
            return None
        if attr not in self.notes_by_id[note_id].keys():
            raise KeyError("Matcher found unknown key")
        return self.notes_by_id[note_id].get(attr, None)

    def get_notes_in_chord(self, note_id):
        """Retrieve all notes in the same chord as the given note ID."""
        note = self.get_note_by_id(note_id)
        if note:
            return [self.get_note_by_id(nid) for nid in note['chord']]
        return None


class MatchNote:
    def __init__(self, onset_time, offset_time, pitch, onset_velocity, offset_velocity, channel, match_status,
                 score_time, note_id, error_index, skip_index):
        self.onset_time = onset_time
        self.offset_time = offset_time
        self.pitch = pitch
        self.onset_velocity = onset_velocity
        self.offset_velocity = offset_velocity
        self.channel = channel
        self.match_status = match_status
        self.score_time = score_time
        self.note_id = note_id
        self.error_index = error_index
        self.skip_index = skip_index


class MissingNote:
    def __init__(self, score_beat, note_id):
        self.score_beat = score_beat
        self.note_id = note_id


class MatchFileParser:
    def __init__(self):
        self.matched_notes = {}  # Dictionary to store notes by onset time (ID)
        self.extra_notes = []  # Dictionary to store extra notes (unaligned pitches)
        self.sorted_notes = []  # (note_id, score_time)
        self.sorted_match_notes = []  # (note_id, score_time)
        self.missing_notes = []  # List to store missing notes
        self.score = ''
        self.perf = ''
        self.fmt3x = ''
        self.version = ''

    def parse_line(self, line):
        # Check if the line starts with //Missing
        line = line.replace('\n', '')
        if line.startswith("//Missing"):
            # Example: //Missing 330 P1-4-42
            parts = line.split()
            score_beat = float(parts[1])  # Parse the score beat
            note_id = parts[2]  # Parse the note ID
            self.missing_notes.append((note_id, score_beat))
            return
        elif line.startswith(('//Version')):
            self.version = line.split(': ')[-1]
            return
        elif line.startswith('// Score'):
            self.score = line.split(': ')[-1]
            return
        elif line.startswith('// Perfm'):
            self.perf = line.split(': ')[-1]
            return
        elif line.startswith('// fmt3x:'):
            self.fmt3x = line.split(': ')[-1]
            return
        elif '//' in line:
            print("\x1B[34m[Info]\033[0m ", line)
            return
        else:
            # Split regular lines by tab and parse the attributes
            parts = line.strip().split("\t")
            try:
                note = {
                    'id': int(parts[0]),
                    'onset_time': float(parts[1]),  # Onset time (ID)
                    'offset_time': float(parts[2]),  # Offset time
                    'pitch': parts[3],  # Spelled pitch
                    'onset_velocity': int(parts[4]),  # Onset velocity
                    'offset_velocity': int(parts[5]),  # Offset velocity
                    'channel': int(parts[6]),  # Channel
                    'match_status': parts[7],  # Match status
                    'score_time': float(parts[8]),  # Score time
                    'note_id': parts[9],  # Note ID
                    'error_index': int(parts[10]),  # Error index
                    'skip_index': parts[11],  # Skip index
                }
                if parts[9] == '*':
                    self.extra_notes.append(note)
                else:
                    # Store the note in the dictionary indexed by onset_time (ID)
                    self.matched_notes[parts[9]] = note
                    self.sorted_match_notes.append(note)
                self.sorted_notes.append((note['note_id'], note['score_time']))
            except ValueError as e:
                print("A Match File is expected. Format error.")

    def parse_file(self, filepath):
        with open(filepath, "r") as file:
            for line in file:
                self.parse_line(line)

    def count_aligned_midi(self):
        if not self.matched_notes:
            print("\x1B[33m[Warning]\033[0m No match file provided")
            return 0.0
        matched_count = 0
        for note in self.matched_notes.values():
            if note['note_id'] != '*' and note['error_index'] == 0:
                matched_count += 1
        return matched_count

    def get_note_by_onset_time(self, onset_time):
        """Retrieve a note by its onset time (ID)."""
        return self.matched_notes.get(onset_time, None)

    def get_attr_by_id(self, note_id, attr):
        if note_id not in self.matched_notes.keys():
            return None
        if attr not in self.matched_notes[note_id].keys():
            raise KeyError("Matcher found unknown key")
        return self.matched_notes[note_id].get(attr, None)

    def get_onset_time_by_id(self, note_id):
        return self.matched_notes[note_id]['onset_time']

    def get_missing_notes(self):
        """Retrieve all missing notes."""
        return self.missing_notes


class SprParser:
    def __init__(self):
        self.notes_by_id = {}  # Dictionary to store notes by onset time (ID)
        self.sorted_notes = []  # (raw id, onset time)
        self.version = ''

    def parse_file(self, filepath):
        with open(filepath, "r") as file:
            for line in file:
                self.parse_line(line)

    def parse_line(self, line):
        # Check if the line starts with //Missing
        line = line.replace('\n', '')
        if line.startswith('//Version'):
            self.version = line.split(': ')[-1]
            return
        elif '//' in line or line.startswith('#'):
            print("\x1B[34m[Info]\033[0m ", line)
            return
        else:
            # Split regular lines by tab and parse the attributes
            parts = line.strip().split("\t")
            note = {
                'id': int(parts[0]),
                'onset_time': float(parts[1]),  # Onset time (ID)
                'offset_time': float(parts[2]),  # Offset time
                'pitch': parts[3],  # Spelled pitch
                'onset_velocity': int(parts[4]),  # Onset velocity
                'offset_velocity': int(parts[5]),  # Offset velocity
                'channel': int(parts[6]),  # Channel
            }
            # Store the note in the dictionary indexed by onset_time (ID)
            self.notes_by_id[parts[0]] = note
            self.sorted_notes.append(note)  # Assumes already sorted


if __name__ == '__main__':
    import json

    # Example usage:
    parser = ScoreParser()
    parser.parse_file("score_data.txt")

    print(json.dumps(parser.notes_by_id, indent=4))
    print('===================')

    parser = MatchFileParser()
    parser.parse_file("match_data.txt")
    print(json.dumps(parser.matched_notes, indent=4))
    print('===================')

    # note = parser.get_note_by_id("P1-1-2")
    # if note:
    #     print("Note ID:", "P1-1-2")
    #     print("Pitch:", note['pitch'])
    #     print("Duration:", note['duration'])
    #     print("Score Beat:", note['beat'])
    #
    # chord_notes = parser.get_notes_in_chord("P1-1-2")
    # if chord_notes:
    #     print("\nOther notes in the chord:")
    #     for n in chord_notes:
    #         print(f"Note ID: {n['pitch']} (Duration: {n['duration']})")
