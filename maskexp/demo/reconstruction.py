from maskexp.test.test import render_seq, render_contextual_seq

if __name__ == '__main__':
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
