# Interactive Demo

There are mainly two ways to interact with the model.

1. Reconstruction: reconstruct the original MIDI performance given masked input.
2. Prediction: input a melody line as the condition and ask the model to render the rest.

## Reconstruction

TODO @Bmois

## Prediction

This mode requires a musical score (musicxml) and an offline MIDI alignment algorithm.
Many thanks to the following paper introducing a C++ tool for offline MIDI alignment.

> Eita Nakamura, Kazuyoshi Yoshii, Haruhiro Katayose
> Performance Error Detection and Post-Processing for Fast and Accurate Symbolic Music Alignment
> In Proc. ISMIR, pp. 347-353, 2017.

Notice:

- The MIDI alignment feature is not yet robust. Complex score may not work (e.g. with multirests, multiple parts)
- For advanced users, you may additionally include notes other than the melody line to further hint the model.
- Generation of sustain pedal markings is not yet supported. You may manually add them after prediction. 

### Quickstart
Input format:
- SPATH: ends with .xml
- PPATH: ends with .mid
- RPATH: ends with .mid

```shell
python prediction.py  \
        --score_path=SPATH  \
        --performance_path=PPATH  \
        # --ref_midi_path=RPATH \
        --output_dir=DIR  \
        --file_stem="test"
```