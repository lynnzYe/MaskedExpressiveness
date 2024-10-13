from maskexp.test.test import render_seq, render_contextual_seq
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reconstruct expressive performance given masked MIDI input'
    )
    # Adding arguments
    parser.add_argument('--midi_path', type=str, required=True, help="MIDI file path")
    parser.add_argument('--model_path', type=str, required=True, help="checkpoint file path")
    parser.add_argument('--mask_mode', type=str, required=True,
                        help="Mode for masking: all | some | min | none \n"
                             "all: mask all velocity information and have model reconstruct from scratch\n"
                             "some: randomly mask some velocity information"
                             "min: assume that the input MIDI file is already masked (velocity set to 1 for a mask)"
                             'none: do nothing')
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where output will be saved")
    parser.add_argument('--file_stem', type=str, required=True, help="The file stem for output file naming")

    midi_path = '/Users/kurono/Desktop/AlignmentTool/test.mid'
    ckpt_path = '/Users/kurono/Documents/python/GEC/ExpressiveMLM/save/checkpoints/kg_rawmlm.pth'
    mask_mode = 'min'

    render_seq(midi_path=midi_path,
               ckpt_path=ckpt_path,
               mask_mode=mask_mode)

    render_contextual_seq(
        midi_path=midi_path,
        ckpt_path=ckpt_path,
        mask_mode=mask_mode,
        file_stem='contextdemo1')
