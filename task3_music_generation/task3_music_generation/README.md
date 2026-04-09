# Task 3: Music Generation with AI

## Overview

An end-to-end pipeline that:
1. Collects classical / jazz MIDI data (built-in corpus or your own files)
2. Preprocesses notes into integer-encoded sequences via music21
3. Trains a stacked LSTM model in TensorFlow/Keras
4. Generates new note sequences with temperature-controlled sampling
5. Exports the result as a playable .mid file

---

## Project Structure

```
task3_music_generation/
    music_generator.py   - Main pipeline (collect, preprocess, train, generate, export)
    midi_utils.py        - MIDI inspection, augmentation, playback helpers
    requirements.txt     - Python dependencies
    data/                - Auto-created; stores the note-token pickle cache
    checkpoints/         - Auto-created; stores best model weights during training
    output/              - Auto-created; stores generated MIDI files
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with built-in corpus (no extra files needed)
```bash
python music_generator.py
```
This fetches Bach, Beethoven, Mozart, and Joplin from music21's built-in corpus.

### 3. Run with your own MIDI files
```bash
python music_generator.py --midi_dir /path/to/your/midi/folder
```

### 4. Generate only (using pre-trained weights)
```bash
python music_generator.py \
    --load_weights checkpoints/best_weights.h5 \
    --skip_training \
    --generation_length 300 \
    --temperature 0.9
```

### 5. Full example with all options
```bash
python music_generator.py \
    --midi_dir ./my_midi_files \
    --epochs 50 \
    --generation_length 200 \
    --temperature 1.0 \
    --checkpoint_dir ./checkpoints \
    --output_dir ./output
```

---

## CLI Arguments

| Argument            | Default                    | Description                                         |
|---------------------|----------------------------|-----------------------------------------------------|
| `--midi_dir`        | None (use corpus)          | Directory of .mid files                             |
| `--epochs`          | 100                        | Maximum training epochs                             |
| `--generation_length` | 200                      | Number of notes to generate                         |
| `--temperature`     | 1.0                        | Sampling temperature (0.5=safe, 1.0=balanced, 1.5=creative) |
| `--load_weights`    | None                       | Path to pre-trained .h5 weights                     |
| `--skip_training`   | False                      | Skip training, go straight to generation            |
| `--cache_path`      | data/notes_cache.pkl       | Note-token cache path                               |
| `--checkpoint_dir`  | checkpoints/               | Directory for model checkpoints                     |
| `--output_dir`      | output/                    | Directory for generated MIDI                        |

---

## Playing the Generated MIDI

Open `output/generated_music.mid` with any of:
- **VLC Media Player** (cross-platform, free)
- **GarageBand** (macOS/iOS)
- **MuseScore** (also displays sheet music)
- **Windows Media Player**

### Optional: Convert to WAV
Install FluidSynth and a soundfont:
```bash
# Ubuntu/Debian
sudo apt-get install fluidsynth fluid-soundfont-gm

# macOS
brew install fluid-synth
```
Then from Python:
```python
from midi_utils import convert_midi_to_wav_fluidsynth
convert_midi_to_wav_fluidsynth("output/generated_music.mid")
```

---

## Model Architecture

```
Input: (batch, sequence_length=100, 1)
  |
LSTM(512, return_sequences=True) -> BatchNorm -> Dropout(0.3)
  |
LSTM(256, return_sequences=True) -> BatchNorm -> Dropout(0.3)
  |
LSTM(128)                        -> BatchNorm -> Dropout(0.3)
  |
Dense(256) -> ReLU -> Dropout(0.3)
  |
Dense(vocab_size) -> Softmax
  |
Output: probability over all unique note/chord tokens
```

Training callbacks:
- **ModelCheckpoint** - saves the best weights by training loss
- **EarlyStopping**   - stops training when loss stops improving (patience=10)
- **ReduceLROnPlateau** - halves the LR when loss plateaus (patience=5)

---

## Note Representation

Each note is encoded as one of three token types:

| Type   | Example     | Description                          |
|--------|-------------|--------------------------------------|
| Pitch  | `C4`        | Single note with octave              |
| Chord  | `0.4.7`     | Chord as dot-separated normal-order  |
| Rest   | `R`         | Silence                              |

Tokens are integer-encoded, normalized to [0,1], and fed as sequences of length 100.

---

## Data Augmentation

Use `midi_utils.augment_dataset()` to multiply training data via transposition:
```python
from midi_utils import augment_dataset
augmented_notes = augment_dataset(notes, n_augmentations=4, semitone_range=(-6, 6))
```

---

## Tips for Better Results

- Use at least 10-20 MIDI files from the same genre for coherent output.
- Train for at least 50 epochs; loss typically drops significantly in the first 20.
- Lower temperature (0.5-0.8) for more structured output.
- Higher temperature (1.2-1.5) for more experimental / creative output.
- The cache file (`data/notes_cache.pkl`) is reused on subsequent runs; delete it
  if you change the input MIDI files.
