from pathlib import Path
import note_seq

ROOT_DIR = str(Path(__file__).parent.parent)
OUTPUT_DIR = ROOT_DIR + '/outputs'
DATA_DIR = ROOT_DIR + '/data'
SAVE_DIR = ROOT_DIR + '/save'

NDEBUG = False

IGNORE_LABEL_INDEX = -100
# Piano has midi range 21 ~ 108, while magenta encodes 128 pitches. Therefore, we use the unused pitches for mask tokens
DEFAULT_MASK_EVENT = note_seq.PerformanceEvent(event_type=1, event_value=1)
VELOCITY_MASK_EVENT = note_seq.PerformanceEvent(event_type=1, event_value=2)

if __name__ == '__main__':
    print(ROOT_DIR)
