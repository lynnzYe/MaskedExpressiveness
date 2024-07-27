from pathlib import Path

ROOT_DIR = str(Path(__file__).parent.parent)
OUTPUT_DIR = ROOT_DIR + '/outputs'
DATA_DIR = ROOT_DIR + '/data'

if __name__ == '__main__':
    print(ROOT_DIR)
