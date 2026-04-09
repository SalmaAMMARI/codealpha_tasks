"""
midi_utils.py - MIDI inspection, augmentation, and optional audio playback utilities.

This module is kept separate from the main pipeline so it can be imported
without requiring pygame (which needs a display).
"""

import os
import random
import numpy as np

try:
    from music21 import converter, tempo, key, meter, stream, note, chord, instrument
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


# ===========================================================================
# MIDI INSPECTION
# ===========================================================================

def inspect_midi(file_path):
    """
    Print a human-readable summary of a MIDI file.

    Parameters
    ----------
    file_path : str
        Path to the MIDI or MusicXML file.
    """
    if not MUSIC21_AVAILABLE:
        raise ImportError("music21 is required.")

    midi = converter.parse(file_path)
    print(f"--- MIDI Inspection: {os.path.basename(file_path)} ---")
    print(f"  Parts        : {len(midi.parts)}")
    print(f"  Total notes  : {len(midi.flat.notes)}")

    tempos = midi.flat.getElementsByClass(tempo.MetronomeMark)
    if tempos:
        print(f"  Tempo        : {tempos[0].number} BPM")

    keys = midi.analyze("key")
    print(f"  Estimated key: {keys}")

    time_sigs = midi.flat.getElementsByClass(meter.TimeSignature)
    if time_sigs:
        print(f"  Time signature: {time_sigs[0]}")
    print()


# ===========================================================================
# DATA AUGMENTATION
# ===========================================================================

def transpose_note_list(notes, semitones):
    """
    Transpose all pitch tokens in a note list by a fixed number of semitones.

    Chord tokens (dot-separated integers) and rest tokens are handled correctly.
    Pitch tokens that cannot be transposed are passed through unchanged.

    Parameters
    ----------
    notes : list of str
        Note/chord/rest tokens.
    semitones : int
        Number of semitones to shift (positive = up, negative = down).

    Returns
    -------
    list of str
        Transposed token list.
    """
    if not MUSIC21_AVAILABLE:
        raise ImportError("music21 is required.")

    transposed = []
    for token in notes:
        if token == "R":
            transposed.append(token)
        elif "." in token:
            # Chord: shift each integer (normal-order pitch class)
            try:
                shifted = ".".join(
                    str((int(n) + semitones) % 12) for n in token.split(".")
                )
                transposed.append(shifted)
            except ValueError:
                transposed.append(token)
        else:
            try:
                n = note.Note(token)
                n.transpose(semitones, inPlace=True)
                transposed.append(str(n.pitch))
            except Exception:
                transposed.append(token)

    return transposed


def augment_dataset(notes, n_augmentations=4, semitone_range=(-6, 6)):
    """
    Augment a note token list by random transposition.

    Parameters
    ----------
    notes : list of str
        Original note token list.
    n_augmentations : int
        Number of transposed copies to add.
    semitone_range : tuple of (int, int)
        Range of semitone shifts to sample from.

    Returns
    -------
    list of str
        Original + augmented tokens concatenated.
    """
    augmented = list(notes)
    for _ in range(n_augmentations):
        shift = random.randint(*semitone_range)
        augmented.extend(transpose_note_list(notes, shift))
    print(f"[AUGMENT] Dataset size: {len(notes)} -> {len(augmented)} tokens after augmentation.")
    return augmented


# ===========================================================================
# STATISTICS
# ===========================================================================

def compute_note_statistics(notes):
    """
    Compute and print basic statistics about a note token list.

    Parameters
    ----------
    notes : list of str
        Note/chord/rest token list.
    """
    from collections import Counter

    total   = len(notes)
    counts  = Counter(notes)
    rests   = counts.get("R", 0)
    chords  = sum(1 for t in notes if "." in t)
    pitches = total - rests - chords

    print("--- Note Statistics ---")
    print(f"  Total tokens : {total}")
    print(f"  Unique tokens: {len(counts)}")
    print(f"  Single pitches: {pitches} ({100*pitches/total:.1f}%)")
    print(f"  Chords        : {chords} ({100*chords/total:.1f}%)")
    print(f"  Rests         : {rests} ({100*rests/total:.1f}%)")
    print(f"  Top-10 tokens : {counts.most_common(10)}")
    print()


# ===========================================================================
# OPTIONAL PLAYBACK
# ===========================================================================

def play_midi_pygame(midi_path):
    """
    Play a MIDI file using pygame.mixer (requires pygame and a MIDI soundfont).

    Parameters
    ----------
    midi_path : str
        Path to the .mid file.
    """
    try:
        import pygame
    except ImportError:
        print("[PLAY] pygame is not installed. Install with: pip install pygame")
        print(f"       Open '{midi_path}' manually in a MIDI player (VLC, GarageBand, etc.).")
        return

    pygame.init()
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(midi_path)
        pygame.mixer.music.play()
        print(f"[PLAY] Playing '{midi_path}' ... (press Ctrl-C to stop)")
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as exc:
        print(f"[PLAY] Playback failed: {exc}")
    finally:
        pygame.mixer.quit()
        pygame.quit()


def convert_midi_to_wav_fluidsynth(midi_path, wav_path=None, soundfont=None):
    """
    Convert a MIDI file to WAV using FluidSynth (must be installed separately).

    FluidSynth installation:
      Ubuntu/Debian : sudo apt-get install fluidsynth
      macOS         : brew install fluid-synth
      Windows       : https://www.fluidsynth.org/

    A General MIDI soundfont is also required, e.g.:
      sudo apt-get install fluid-soundfont-gm
      Default path  : /usr/share/sounds/sf2/FluidR3_GM.sf2

    Parameters
    ----------
    midi_path : str
        Path to the input .mid file.
    wav_path : str or None
        Output .wav path. Defaults to same name as input with .wav extension.
    soundfont : str or None
        Path to a .sf2 soundfont. Uses a common default if None.

    Returns
    -------
    str or None
        Path to the created WAV file, or None on failure.
    """
    if wav_path is None:
        wav_path = os.path.splitext(midi_path)[0] + ".wav"

    if soundfont is None:
        candidates = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            "/usr/local/share/fluidsynth/soundfonts/GeneralUser.sf2",
        ]
        soundfont = next((p for p in candidates if os.path.exists(p)), None)

    if soundfont is None:
        print("[EXPORT] No soundfont found. Provide a .sf2 file path via the soundfont parameter.")
        return None

    cmd = f'fluidsynth -ni "{soundfont}" "{midi_path}" -F "{wav_path}" -r 44100'
    ret = os.system(cmd)
    if ret == 0:
        print(f"[EXPORT] WAV saved to: {os.path.abspath(wav_path)}")
        return wav_path
    else:
        print("[EXPORT] FluidSynth conversion failed. Is fluidsynth installed?")
        return None
