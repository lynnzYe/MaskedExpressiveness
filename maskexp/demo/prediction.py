import os.path
import shutil
from pathlib import Path
import argparse

from pandas.io.pytables import performance_doc

from maskexp.test.test import render_seq, render_contextual_seq
from maskexp.util.alignment_parser import ScoreParser, MatchFileParser, SprParser
from maskexp.util.midifier import find_missing_midi, write_coop_midi
from maskexp.definitions import OUTPUT_DIR, SAVE_DIR


def midify(fmt3x_file, match_file, spr_file, filestem='perf_pred'):
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
    match_info.parse_file(match_file)

    if spr_file is not None:
        spr_info = SprParser()
        spr_info.parse_file(spr_file)
        success_rate = match_info.count_aligned_midi() / len(spr_info.sorted_notes)
        print("Alignment success rate:", success_rate)
    else:
        spr_info = None
        success_rate = match_info.count_aligned_midi() / len(match_info.sorted_notes)
        print("Alignment success rate:", success_rate)
        if success_rate < 0.8:
            print("\x1B[33m[Warning]\033[0m Poorly Aligned. "
                  "You may want to try MIDI-MIDI alignment instead of score alignment. "
                  "Convert the xml to MIDI using Musescore, for example")

    med_midi_list, midi_list = find_missing_midi(score_info, match_info, spr_info)
    write_coop_midi(med_midi_list, midi_list, os.path.join(OUTPUT_DIR, filestem + '.mid'))


def build_alignment_tool():
    og_dir = os.getcwd()
    os.chdir('AlignmentTool')
    os.system('./compile.sh')
    os.chdir(og_dir)


def run_score_alignment(score, midi):
    """
    Run AlignmentTool for score-midi offline Alignment
    :param score: relative file path with no ext
    :param midi: relative file path with no ext
    :return:
    """
    og_dir = os.getcwd()
    os.chdir('AlignmentTool')
    os.system(f'./MusicXMLToMIDIAlign.sh {score} {midi}')
    os.chdir(og_dir)


def run_midi_alignment(ref_midi, match_midi):
    """
    MIDI to MIDI alignment
    :param ref_midi:     the MIDI as the reference for matching
                            this is usually the user input, because empirically
                            matching a dense MIDI to a sparse performance performs better than the reverse
    :param match_midi:   the MIDI to be matched
    :return:
    """
    og_dir = os.getcwd()
    os.chdir('AlignmentTool')
    os.system(f'./MIDIToMIDIAlign.sh {ref_midi} {match_midi}')
    os.chdir(og_dir)


def pred_performance(performance_path, score_path, score_midi_path, output_dir, file_stem):
    """
    Predict the velocity of missing note events given an initial condition.

    :param performance_path: absolute path to the performance (midi file). same as above
    :param score_path: absolute path to the score. will be copied and stored in AlignmentTool/data
    :param score_midi_path: absolute path to the score MIDI
                            use this mode (MIDI alignment) when score alignment does not work well
    :param output_dir:
    :param file_stem:
    :return:
    """
    assert (score_path is None) != (score_midi_path is None)
    if not os.path.exists('AlignmentTool/Programs'):
        build_alignment_tool()
    Path('AlignmentTool/data').mkdir(exist_ok=True)
    shutil.copy(performance_path, 'AlignmentTool/data')
    perf_file = os.path.join('data', os.path.basename(performance_path).split('.')[0])

    if score_path is None:
        shutil.copy(score_midi_path, 'AlignmentTool/data')
        score_midi = os.path.join('data', os.path.basename(score_midi_path).split('.')[0])
        run_midi_alignment(perf_file, score_midi)

        fmt3x_file = os.path.join('AlignmentTool', perf_file + "_fmt3x.txt")
        match_file = os.path.join('AlignmentTool', score_midi + '_match.txt')
        spr_file = os.path.join('AlignmentTool', perf_file + '_spr.txt')
        midify(fmt3x_file, match_file, spr_file, filestem=file_stem)

    else:
        shutil.copy(score_path, 'AlignmentTool/data')
        score_file = os.path.join('data', os.path.basename(score_path).split('.')[0])
        run_score_alignment(score_file, perf_file)

        fmt3x_file = os.path.join('AlignmentTool', score_file + "_fmt3x.txt")
        match_file = os.path.join('AlignmentTool', perf_file + '_match.txt')
        midify(fmt3x_file, match_file, None, filestem=file_stem)

    midi_path = os.path.join(OUTPUT_DIR, file_stem + '.mid')
    ckpt_path = os.path.join(SAVE_DIR, 'checkpoints', 'kg_rawmlm.pth')
    render_contextual_seq(midi_path, ckpt_path, mask_mode='min', output_path=output_dir, file_stem=file_stem)
    # render_seq(midi_path, ckpt_path, mask_mode='min', output_path=output_dir, file_stem=file_stem)


def test_run_alignment():
    score_path = '/Users/kurono/Desktop/schu.xml'
    performance_path = '/Users/kurono/Desktop/schu.mid'
    ref_midi = '/Users/kurono/Desktop/schu_score.mid'

    shutil.copy(score_path, 'AlignmentTool/data')
    shutil.copy(performance_path, 'AlignmentTool/data')
    shutil.copy(ref_midi, 'AlignmentTool/data')

    score_file = os.path.join('data', os.path.basename(score_path).split('.')[0])
    midi_file = os.path.join('data', os.path.basename(performance_path).split('.')[0])
    ref_midi_file = os.path.join('data', os.path.basename(ref_midi).split('.')[0])

    run_score_alignment(score_file, midi_file)
    match_file = os.path.join('AlignmentTool', midi_file + '_match.txt')

    # run_midi_alignment(midi_file, ref_midi_file)
    # run_midi_alignment(ref_midi_file, midi_file)
    # match_file = os.path.join('AlignmentTool', ref_midi_file + '_match.txt')
    # score_file = os.path.join('AlignmentTool', midi_file + '_fmt3x.txt')
    #
    # score_parser = ScoreParser()
    # score_parser.parse_file(score_file)

    parser = MatchFileParser()
    parser.parse_file(match_file)

    # target_midi_count = len(score_parser.sorted_notes)
    target_midi_count = len(parser.sorted_notes)
    print('Alignment Success Rate: ', parser.count_aligned_midi() / target_midi_count)


def test_main():
    # score_path = '/Users/kurono/Desktop/Chop.xml'
    score_path = '/Users/kurono/Desktop/schu.xml'
    # performance_path = '/Users/kurono/Desktop/Chop.mid'
    performance_path = '/Users/kurono/Desktop/schu.mid'
    ref_midi_path = '/Users/kurono/Desktop/schu_score.mid'
    output_dir = '/Users/kurono/Desktop'
    file_stem = 'test_main'

    # pred_performance(performance_path, score_path, None, output_dir, file_stem)
    pred_performance(performance_path, None, ref_midi_path, output_dir, file_stem)


def main():
    parser = argparse.ArgumentParser(
        description='Predict expressive performance by conditioning on performed parts of the score'
    )
    # Adding arguments
    parser.add_argument('--score_path', type=str, required=True, help="Path to the score file")
    parser.add_argument('--performance_path', type=str, required=False, help="Path to the performance file")
    parser.add_argument('--ref_midi_path', type=str, required=False, help="Path to the performance file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where output will be saved")
    parser.add_argument('--file_stem', type=str, required=True, help="The file stem for output file naming")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_performance(args.performance_path, args.score_path, args.ref_midi_path, args.output_dir, args.file_stem)


if __name__ == '__main__':
    # Sometimes you may notice the model predicting the same velocity for continuous notes.
    # This is because the note_seq lib uses one velocity token to describe several notes with the same velocity.
    # (which is not ideal in our case)
    # To solve this, you need to manually modify line 379 in note_seq/performance_lib.py
    # - if not is_offset and velocity_bin != current_velocity_bin:
    # + if not is_offset:
    # TODO @Bmois move this to Readme.md

    # test_run_alignment()
    test_main()
    # main()
