from note_seq import midi_io
import pretty_midi as pm


def parse_a_midi(file):
    seq = midi_io.midi_to_note_sequence(pm.PrettyMIDI(file))
    seq.notes
    return seq


if __name__ == '__main__':
    print("Tokenize midi")
    test = parse_a_midi('../data/ATEPP-1.2-cleaned/Sergei_Rachmaninoff/Variations_on_a_Theme_of_Chopin/Theme/00077.mid')
    print(test)
