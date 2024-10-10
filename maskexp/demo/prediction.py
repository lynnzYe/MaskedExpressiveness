from maskexp.test.test import render_seq
from maskexp.util.alignment_parser import ScoreParser, MatchFileParser
from maskexp.util.midifier import find_missing_midi


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


if __name__ == '__main__':
    pass
