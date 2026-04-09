"""
Task 3: Music Generation with AI
Uses LSTM-based deep learning to learn and generate MIDI music sequences.
Supports classical/jazz MIDI data collection, preprocessing, training, and generation.
"""

import os
import sys
import glob
import pickle
import random
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Optional heavy imports - guarded so the module can be imported for testing
# ---------------------------------------------------------------------------
try:
    from music21 import corpus, converter, instrument, note, chord, stream, pitch
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Activation
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH   = 100      # number of notes used as input context
BATCH_SIZE        = 64
EPOCHS            = 100
LEARNING_RATE     = 0.001
LSTM_UNITS_1      = 512
LSTM_UNITS_2      = 256
DROPOUT_RATE      = 0.3
GENERATION_LENGTH = 200      # notes to generate per output piece
TEMPERATURE       = 1.0      # sampling temperature (higher = more random)


# ===========================================================================
# 1. DATA COLLECTION
# ===========================================================================

def collect_midi_from_corpus(genres=("classical", "jazz"), max_pieces=50):
    """
    Collect MIDI paths from the music21 built-in corpus.

    Parameters
    ----------
    genres : tuple of str
        Composer/genre tags to search inside the corpus.
    max_pieces : int
        Maximum number of pieces to collect.

    Returns
    -------
    list of str
        Paths to MIDI / MXL files found in the corpus.
    """
    if not MUSIC21_AVAILABLE:
        raise ImportError("music21 is required. Install with: pip install music21")

    paths = []
    composer_map = {
        "classical": ["bach", "beethoven", "mozart", "schubert", "handel"],
        "jazz":      ["joplin"],          # ragtime in corpus; closest to jazz
    }

    for genre in genres:
        composers = composer_map.get(genre, [genre])
        for composer in composers:
            found = corpus.getComposer(composer)
            paths.extend(found)
            if len(paths) >= max_pieces:
                break
        if len(paths) >= max_pieces:
            break

    paths = list(set(paths))[:max_pieces]
    print(f"[DATA] Collected {len(paths)} pieces from corpus.")
    return paths


def collect_midi_from_directory(directory):
    """
    Collect all MIDI files from a user-supplied directory.

    Parameters
    ----------
    directory : str
        Root directory to search recursively.

    Returns
    -------
    list of str
        Absolute paths to .mid / .midi files found.
    """
    patterns = ["**/*.mid", "**/*.midi", "**/*.MID", "**/*.MIDI"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(directory, pat), recursive=True))
    paths = list(set(paths))
    print(f"[DATA] Found {len(paths)} MIDI files in '{directory}'.")
    return paths


# ===========================================================================
# 2. PREPROCESSING
# ===========================================================================

def parse_midi_to_notes(file_path):
    """
    Parse a single MIDI / MXL file and extract a flat list of note tokens.

    Each token is one of:
      - A pitch string, e.g. "C4", "G#3"
      - A chord string, e.g. "C4.E4.G4"
      - A rest token "R"

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the MIDI or MusicXML file.

    Returns
    -------
    list of str
        Ordered sequence of note/chord/rest tokens.
    """
    if not MUSIC21_AVAILABLE:
        raise ImportError("music21 is required.")

    notes_out = []
    try:
        midi = converter.parse(file_path)
    except Exception as exc:
        print(f"  [WARN] Could not parse {file_path}: {exc}")
        return notes_out

    # Flatten to a single part for simplicity
    try:
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes
    except Exception:
        elements = midi.flat.notes

    for element in elements:
        if isinstance(element, note.Note):
            notes_out.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            token = ".".join(str(n) for n in element.normalOrder)
            notes_out.append(token)
        elif isinstance(element, note.Rest):
            notes_out.append("R")

    return notes_out


def preprocess_midi_files(file_paths, cache_path="data/notes_cache.pkl"):
    """
    Parse all MIDI files and return a combined list of note tokens.
    Results are cached to disk so repeated runs are fast.

    Parameters
    ----------
    file_paths : list
        Paths to MIDI files.
    cache_path : str
        Where to store / load the pickle cache.

    Returns
    -------
    list of str
        All note tokens from all files concatenated.
    """
    if os.path.exists(cache_path):
        print(f"[PREPROCESS] Loading cached notes from '{cache_path}'.")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    all_notes = []
    for i, path in enumerate(file_paths):
        print(f"  Parsing file {i+1}/{len(file_paths)}: {os.path.basename(str(path))}")
        all_notes.extend(parse_midi_to_notes(path))

    with open(cache_path, "wb") as f:
        pickle.dump(all_notes, f)

    print(f"[PREPROCESS] Total note tokens extracted: {len(all_notes)}")
    return all_notes


def build_sequences(notes, sequence_length=SEQUENCE_LENGTH):
    """
    Convert the flat note list into (input_sequence, target_note) pairs
    and encode everything as integers.

    Parameters
    ----------
    notes : list of str
        Flat list of note/chord/rest tokens.
    sequence_length : int
        Length of each input window.

    Returns
    -------
    X : np.ndarray, shape (N, sequence_length, 1)
        Normalized integer-encoded input sequences.
    y : np.ndarray, shape (N, vocab_size)
        One-hot encoded target notes.
    note_to_int : dict
        Mapping from token string to integer index.
    int_to_note : dict
        Reverse mapping.
    vocab_size : int
        Number of unique note tokens.
    """
    unique_notes = sorted(set(notes))
    vocab_size   = len(unique_notes)
    note_to_int  = {n: i for i, n in enumerate(unique_notes)}
    int_to_note  = {i: n for n, i in note_to_int.items()}

    network_input  = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in  = notes[i : i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)

    # Reshape and normalize for LSTM input
    X = np.reshape(network_input, (n_patterns, sequence_length, 1))
    X = X / float(vocab_size)                          # normalize to [0, 1]
    y = to_categorical(network_output, num_classes=vocab_size)

    print(f"[SEQUENCES] {n_patterns} training pairs | vocab size: {vocab_size}")
    return X, y, note_to_int, int_to_note, vocab_size


# ===========================================================================
# 3. MODEL DEFINITION
# ===========================================================================

def build_lstm_model(sequence_length, vocab_size, learning_rate=LEARNING_RATE):
    """
    Build a stacked LSTM model for note sequence prediction.

    Architecture:
      LSTM(512, return_sequences=True)
      BatchNorm + Dropout
      LSTM(256, return_sequences=True)
      BatchNorm + Dropout
      LSTM(128)
      BatchNorm + Dropout
      Dense(256) + ReLU
      Dropout
      Dense(vocab_size) + Softmax

    Parameters
    ----------
    sequence_length : int
        Length of input windows.
    vocab_size : int
        Number of unique output classes.
    learning_rate : float
        Adam optimizer learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled model ready for training.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    model = Sequential([
        LSTM(LSTM_UNITS_1, input_shape=(sequence_length, 1), return_sequences=True),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        LSTM(LSTM_UNITS_2, return_sequences=True),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        LSTM(128, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),

        Dense(256),
        Activation("relu"),
        Dropout(0.3),

        Dense(vocab_size),
        Activation("softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()
    return model


# ===========================================================================
# 4. TRAINING
# ===========================================================================

def train_model(model, X, y, checkpoint_dir="checkpoints", epochs=EPOCHS):
    """
    Train the LSTM model with checkpointing, early stopping, and LR scheduling.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled model.
    X : np.ndarray
        Training inputs.
    y : np.ndarray
        One-hot training targets.
    checkpoint_dir : str
        Directory to save model checkpoints.
    epochs : int
        Maximum training epochs.

    Returns
    -------
    tf.keras.callbacks.History
        Training history object.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_weights_path = os.path.join(checkpoint_dir, "best_weights.h5")

    callbacks = [
        ModelCheckpoint(
            best_weights_path,
            monitor="loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    print(f"[TRAIN] Best weights saved to: {best_weights_path}")
    return history


# ===========================================================================
# 5. GENERATION
# ===========================================================================

def sample_with_temperature(probabilities, temperature=TEMPERATURE):
    """
    Sample an index from a probability distribution with temperature scaling.

    Lower temperature -> more deterministic (picks highest-probability notes).
    Higher temperature -> more random / creative output.

    Parameters
    ----------
    probabilities : np.ndarray
        Raw softmax output from the model.
    temperature : float
        Scaling factor.

    Returns
    -------
    int
        Sampled index.
    """
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-8) / temperature
    probabilities = np.exp(probabilities)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


def generate_note_sequence(
    model,
    network_input,
    note_to_int,
    int_to_note,
    vocab_size,
    sequence_length=SEQUENCE_LENGTH,
    generation_length=GENERATION_LENGTH,
    temperature=TEMPERATURE,
):
    """
    Generate a sequence of note tokens using the trained model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained LSTM model.
    network_input : np.ndarray
        All training input sequences (used to pick a random seed).
    note_to_int : dict
        Token-to-index mapping.
    int_to_note : dict
        Index-to-token mapping.
    vocab_size : int
        Vocabulary size.
    sequence_length : int
        Context window length.
    generation_length : int
        Number of notes to generate.
    temperature : float
        Sampling temperature.

    Returns
    -------
    list of str
        Generated note/chord/rest tokens.
    """
    # Pick a random seed from the training data
    start_idx = random.randint(0, len(network_input) - 1)
    pattern   = list(network_input[start_idx].flatten() * vocab_size)

    generated = []
    for _ in range(generation_length):
        x = np.reshape(pattern, (1, sequence_length, 1)) / float(vocab_size)
        prediction = model.predict(x, verbose=0)[0]
        index  = sample_with_temperature(prediction, temperature)
        result = int_to_note[index]
        generated.append(result)
        pattern.append(index)
        pattern = pattern[1:]  # slide the window

    return generated


# ===========================================================================
# 6. MIDI CONVERSION & EXPORT
# ===========================================================================

def notes_to_midi(note_tokens, output_path="output/generated_music.mid", tempo=120):
    """
    Convert generated note tokens back into a MIDI file.

    Parameters
    ----------
    note_tokens : list of str
        Tokens produced by generate_note_sequence().
    output_path : str
        Destination path for the MIDI file.
    tempo : int
        BPM for the output piece.

    Returns
    -------
    str
        Absolute path to the saved MIDI file.
    """
    if not MUSIC21_AVAILABLE:
        raise ImportError("music21 is required.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    output_notes = []
    offset = 0.0          # beat position within the score
    duration = 0.5        # default quarter-note duration in beats

    for token in note_tokens:
        if token == "R":
            # Rest
            r = note.Rest()
            r.duration.quarterLength = duration
            r.offset = offset
            output_notes.append(r)

        elif "." in token:
            # Chord - reconstruct from normal-order integers
            try:
                chord_notes = []
                for n_str in token.split("."):
                    new_note = note.Note(int(n_str))
                    new_note.duration.quarterLength = duration
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            except Exception:
                pass

        else:
            # Single note
            try:
                new_note = note.Note(token)
                new_note.duration.quarterLength = duration
                new_note.offset = offset
                output_notes.append(new_note)
            except Exception:
                pass

        offset += duration

    # Build the final stream
    midi_stream = stream.Stream(output_notes)
    midi_stream.insert(0, instrument.Piano())

    midi_stream.write("midi", fp=output_path)
    print(f"[EXPORT] MIDI saved to: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)


# ===========================================================================
# 7. FULL PIPELINE
# ===========================================================================

def run_pipeline(
    midi_dir=None,
    cache_path="data/notes_cache.pkl",
    checkpoint_dir="checkpoints",
    output_dir="output",
    epochs=EPOCHS,
    generation_length=GENERATION_LENGTH,
    temperature=TEMPERATURE,
    load_weights=None,
    skip_training=False,
):
    """
    End-to-end pipeline: collect -> preprocess -> train -> generate -> export.

    Parameters
    ----------
    midi_dir : str or None
        If set, collect MIDI from this directory instead of the corpus.
    cache_path : str
        Path for the note-token cache.
    checkpoint_dir : str
        Directory for model checkpoints.
    output_dir : str
        Directory for generated MIDI files.
    epochs : int
        Training epochs.
    generation_length : int
        Notes to generate.
    temperature : float
        Sampling temperature.
    load_weights : str or None
        If provided, load these weights instead of training from scratch.
    skip_training : bool
        If True and load_weights is set, skip training entirely.
    """
    # -- 1. Collect data
    if midi_dir:
        file_paths = collect_midi_from_directory(midi_dir)
    else:
        file_paths = collect_midi_from_corpus()

    if not file_paths:
        print("[ERROR] No MIDI files found. Provide a directory with --midi_dir.")
        sys.exit(1)

    # -- 2. Preprocess
    notes = preprocess_midi_files(file_paths, cache_path=cache_path)
    if len(notes) < SEQUENCE_LENGTH + 1:
        print("[ERROR] Not enough note data to build sequences. Add more MIDI files.")
        sys.exit(1)

    X, y, note_to_int, int_to_note, vocab_size = build_sequences(notes)

    # -- 3. Build model
    model = build_lstm_model(SEQUENCE_LENGTH, vocab_size)

    if load_weights and os.path.exists(load_weights):
        model.load_weights(load_weights)
        print(f"[MODEL] Loaded weights from: {load_weights}")

    # -- 4. Train
    if not skip_training:
        train_model(model, X, y, checkpoint_dir=checkpoint_dir, epochs=epochs)

    # -- 5. Generate
    print(f"\n[GENERATE] Generating {generation_length} notes at temperature={temperature} ...")
    generated_tokens = generate_note_sequence(
        model, X, note_to_int, int_to_note, vocab_size,
        generation_length=generation_length,
        temperature=temperature,
    )

    # -- 6. Export MIDI
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "generated_music.mid")
    notes_to_midi(generated_tokens, output_path=out_path)

    print("\n[DONE] Pipeline complete.")
    print(f"  Generated MIDI: {os.path.abspath(out_path)}")
    print("  Open the MIDI file with any MIDI player (VLC, GarageBand, MuseScore, etc.)")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LSTM-based MIDI music generation."
    )
    parser.add_argument(
        "--midi_dir", type=str, default=None,
        help="Directory containing user-supplied MIDI files (optional)."
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})."
    )
    parser.add_argument(
        "--generation_length", type=int, default=GENERATION_LENGTH,
        help=f"Notes to generate (default: {GENERATION_LENGTH})."
    )
    parser.add_argument(
        "--temperature", type=float, default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE})."
    )
    parser.add_argument(
        "--load_weights", type=str, default=None,
        help="Path to pre-trained weights (.h5) to load."
    )
    parser.add_argument(
        "--skip_training", action="store_true",
        help="Skip training and go straight to generation (requires --load_weights)."
    )
    parser.add_argument(
        "--cache_path", type=str, default="data/notes_cache.pkl",
        help="Path for the note-token cache file."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory for model checkpoints."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Directory for generated MIDI output."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        midi_dir=args.midi_dir,
        cache_path=args.cache_path,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        generation_length=args.generation_length,
        temperature=args.temperature,
        load_weights=args.load_weights,
        skip_training=args.skip_training,
    )
