import os.path
import shutil
from pathlib import Path
import argparse
from maskexp.test.test import render_seq
from maskexp.util.alignment_parser import ScoreParser, MatchFileParser
from maskexp.util.midifier import find_missing_midi, write_midi
from maskexp.definitions import OUTPUT_DIR, SAVE_DIR


def midify(fmt3x_file, match_file, filestem='perf_pred'):
    """
    Convert a match txt and score txt into MIDI events
    :param fmt3x_file: file path, run AlignmentTool.MusicXMLToMIDIAlign.sh and generate the fmt3x txt
    :param match_file: file path, run AlignmentTool.MusicXMLToMIDIAlign.sh and generate the match txt
    :param filestem:
    :return:
    """
    score_info = ScoreParser()
    score_info.parse_file(fmt3x_file)
    match_info = MatchFileParser()
    match_info.parse_file((match_file))

    midi_list = find_missing_midi(score_info, match_info)
    write_midi(midi_list, os.path.join(OUTPUT_DIR, filestem + '.mid'))


def build_alignment_tool():
    og_dir = os.getcwd()
    os.chdir('AlignmentTool')
    os.system('./compile.sh')
    os.chdir(og_dir)


def run_alignment(midi, score):
    """
    Run AlignmentTool.
    :param midi: relative file path with no ext
    :param score: relative file path with no ext
    :return:
    """
    og_dir = os.getcwd()
    os.chdir('AlignmentTool')
    os.system(f'./MusicXMLToMIDIAlign.sh {score} {midi}')
    os.chdir(og_dir)


def pred_performance(score_path, performance_path, output_dir, file_stem):
    """
    Predict the velocity of missing note events given an initial condition.

    :param score_path: absolute path to the score. will be copied and stored in AlignmentTool/data
    :param performance_path: absolute path to the performance (midi file). same as above
    :param output_dir:
    :param file_stem:
    :return:
    """
    if not os.path.exists('AlignmentTool/Programs'):
        build_alignment_tool()
    Path('AlignmentTool/data').mkdir(exist_ok=True)

    shutil.copy(score_path, 'AlignmentTool/data')
    shutil.copy(performance_path, 'AlignmentTool/data')

    score_file = os.path.join('data', os.path.basename(score_path).split('.')[0])
    midi_file = os.path.join('data', os.path.basename(performance_path).split('.')[0])
    run_alignment(midi_file, score_file)

    fmt3x_file = os.path.join('AlignmentTool', score_file + "_fmt3x.txt")
    match_file = os.path.join('AlignmentTool', midi_file + '_match.txt')
    midi_stem = 'perf_pred'
    midify(fmt3x_file, match_file, filestem=midi_stem)

    midi_path = os.path.join(OUTPUT_DIR, midi_stem + '.mid')
    ckpt_path = os.path.join(SAVE_DIR, 'checkpoints', 'kg_rawmlm.pth')
    render_seq(midi_path, ckpt_path, mask_mode='min', output_path=output_dir, file_stem=file_stem)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict expressive performance by conditioning on performed parts of the score'
    )
    # Adding arguments
    parser.add_argument('--score_path', type=str, required=True, help="Path to the score file")
    parser.add_argument('--performance_path', type=str, required=True, help="Path to the performance file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where output will be saved")
    parser.add_argument('--file_stem', type=str, required=True, help="The file stem for output file naming")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_performance(args.score_path, args.performance_path, args.output_dir, args.file_stem)
