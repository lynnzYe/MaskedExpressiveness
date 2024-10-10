# Interactive Demo

There are mainly two ways to interact with the model.

1. Reconstruction: reconstruct the original MIDI performance given masked input.
2. Prediction: input a melody line as the condition and ask the model to render the rest.

## Reconstruction
TODO @Bmois

## Prediction
To add the missing MIDI notes, this mode requires a musical score (musicxml) and an offline MIDI alignment algorithm.  

Many thanks to the following paper introducing a C++ tool for offline MIDI alignment.

To use the alignment tool, you need to run the `compile.sh` in the AlignmentTool folder.


> Eita Nakamura, Kazuyoshi Yoshii, Haruhiro Katayose
Performance Error Detection and Post-Processing for Fast and Accurate Symbolic Music Alignment
In Proc. ISMIR, pp. 347-353, 2017.