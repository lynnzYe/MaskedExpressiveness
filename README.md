from sympy.physics.units import years

# Masked Expressiveness: Conditioned Generation of Piano Key Striking Velocity Using Masked Language Modeling

---

<u>[Linzan Ye](https://github.com/Bmoist)</u>

> Abstract: Creating and experiencing expressive renditions of composed musical pieces are fundamental to how people
> engage with music. Numerous studies have focused on building computational models of musical expressiveness to gain a
> deeper understanding of its nature and explore potential applications. This paper examines masked language modeling (
> MLM) for modeling expressiveness in piano performance, specifically targeting the prediction of key striking velocity
> using vanilla Bidirectional Encoder Representations from Transformers (BERT). While MLM has been explored in previous
> studies, this work applies it in a novel direction by concentrating on the fine-grained conditioned prediction of
> velocity information. The results show that the model can predict masked velocity events in various contexts within an
> acceptable margin of error, relying solely on the pitch, timing, and velocity data encoded in Musical Instrument
> Digital
> Interface (MIDI) files. Additionally, the study employs a sequential masking and prediction approach toward rendering
> the velocity of unseen MIDI files and achieves more musically convincing results. This approach holds promise for
> developing interactive systems for expressive performance generation, such as advanced piano conducting or
> accompaniment
> systems.

## Installation

```shell
conda create -n maskexp python=3.11 -y
conda activate maskexp
pip install -r requirements.txt
pip install -e .
```

## Interactive Demo

This demo allows user to interact with a pretrained model by inputting user-performed fragments of a musical piece (e.g.
the main melody line). Based on these fragments, the model reconstructs a full expressive performance of the piece,
capturing and extending the user's style, with the help of
an [offline symbolic music alignment algorithm](https://midialignment.github.io/demo.html).

Program Arguments:

- `score_path`: absolute path to the musical score in musicxml format. You will need to rename its extension to `.xml`
  for
  compatibility with the alignment algorithm.
- `performance_path`: absolute path to your performance in the MIDI format (e.g. xxx.mid). Please make sure the
  extension name is `.mid` instead of `.midi`
- `ref_midi_path`: in case that the score is too complex to be aligned well, you may instead provide a MIDI version
  of the
  score. This usually leads to better alignment result, given that the performance is an incomplete fraction of the
  musical piece.
  - if `ref_midi_path` is used, `score_path` should be omitted.
- `output_dir`: directory for writing the predicted performance
- `file_stem`: file name (without extension)

```shell
python maskexp/demo/prediction.py \ 
   --score_path [PATH_TO_XML_SCORE] \
   --performance_path [PATH_TO_MIDI] \
   # --ref_midi_path [PATH_TO_MIDI] \  
   --output_dir [ABSOLUTE_PATH] \
   --file_stem [FILE_NAME]
```

## License

This repository is licensed under the MIT License. However, it includes portions of code from the Magenta project,
which are stored in the magenta folder and licensed under the Apache License 2.0.
Their usage is subject to the terms of the Apache License 2.0.

