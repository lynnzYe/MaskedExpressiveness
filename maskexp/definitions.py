from pathlib import Path
import note_seq

ROOT_DIR = str(Path(__file__).parent.parent)
OUTPUT_DIR = ROOT_DIR + '/outputs'
DATA_DIR = ROOT_DIR + '/data'

IGNORE_LABEL_INDEX = -100
DEFAULT_MASK_EVENT = note_seq.PerformanceEvent(event_type=1, event_value=1)

if __name__ == '__main__':
    print(ROOT_DIR)
