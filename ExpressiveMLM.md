# ExpressiveMLM





# Magenta DAG Pipeline

- create a set of pipelines
  - data processing
    - sustain: convert sustain pedal to sustained notes
    - Quantizer: to 100 (default value for `EventSequenceRnnConfig`)
      - e.g. onset time 1.9 will be 190
      - query by note.quantized_start_step
  - data augmentation
    - stretch
    - transposition
  - data preparation
    - Splitter: break down long sequence into frames/blocks for training
    - PerformanceExtractor: transform to the PerformanceRNN stream
      - NOTEON - TIMESHIFT - NOTEOFF ...
      - quantized at the default value = 100
      - max_shift_step -> time shift is ===, ===, = (combinations of min shifts)
    - encoder: tokenize!
      - PerformanceOneHotEncoding, with velocity bins
      - convert to one hot
- erroneous files:
  - maestro-v3.0.0/2009/MIDI-Unprocessed_03_R1_2009_03-08_ORIG_MID--AUDIO_03_R1_2009_03_R1_2009_08_WAV.midi


## Performance RNN generation:

- input performance object (after `PerformanceExtractor`)
- inside: encoder_decoder converts into one-hot