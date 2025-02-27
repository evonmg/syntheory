"""
Some simple progressions from here:
    - https://www.libertyparkmusic.com/5-common-guitar-chord-progressions/
    - https://www.pianote.com/blog/piano-chord-progressions/
    - https://www.study-guitar.com/blog/minor-key-chord-progressions/
    - https://blog.native-instruments.com/minor-chord-progressions/
    - https://www.justinguitar.com/guitar-lessons/5-common-chord-progressions-bg-1011
"""

import warnings
from typing import Iterator, Tuple, List, Dict, Any, Iterable
from pathlib import Path

from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription
from dataset.music.transforms import get_chord, get_scale
from dataset.music.constants import PITCH_CLASS_TO_NOTE_NAME_SHARP, NOTE_NAME_TO_ENHARMONIC
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_progression,
)
from dataset.synthetic.midi_instrument import get_instruments
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent

PROGRESSIONS = (
    # tuple of mode, scale_degrees
    ("ionian", (1, 4, 5, 1)),
    ("ionian", (1, 4, 6, 5)),
    ("ionian", (1, 5, 6, 4)),
    ("ionian", (1, 6, 4, 5)),
    ("ionian", (2, 5, 1, 6)),
    ("ionian", (4, 1, 5, 6)),
    ("ionian", (4, 5, 3, 6)),
    ("ionian", (5, 4, 1, 5)),
    ("ionian", (5, 6, 4, 1)),
    ("ionian", (6, 4, 1, 5)),
    ("aeolian", (1, 2, 5, 1)),
    ("aeolian", (1, 3, 4, 1)),
    ("aeolian", (1, 4, 5, 1)),
    ("aeolian", (1, 6, 3, 7)),
    ("aeolian", (1, 6, 7, 1)),
    ("aeolian", (1, 6, 7, 3)),
    ("aeolian", (1, 7, 6, 4)),
    ("aeolian", (4, 7, 1, 1)),
    ("aeolian", (7, 6, 7, 1)),
)

_NOTE_TO_RELATIVE_MAJOR = {
    "F#": "A",
    "G": "Bb",
    "G#": "B",
    "A": "C",
    "A#": "C#",
    "Bb": "Db",
    "B": "D",
    "C": "Eb",
    "C#": "E",
    "D": "F",
    "D#": "F#",
    "Eb": "Gb",
    "E": "G",
    "F": "Ab",
}

_NOTE_TO_MAJOR_SCALE = {
    "C": ["C", "D", "E", "F", "G", "A", "B"],
    "C#": ["C#", "D#", "E#", "F#", "G#", "A#", "B#"],
    "Db": ["Db", "Eb", "F", "Gb", "Ab", "Bb", "C"],
    "D": ["D", "E", "F#", "G", "A", "B", "C#"],
    "Eb": ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
    "E": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "F": ["F", "G", "A", "Bb", "C", "D", "E"],
    "F#": ["F#", "G#", "A#", "B", "C#", "D#", "E#"],
    "Gb": ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F"],
    "G": ["G", "A", "B", "C", "D", "E", "F#"],
    "Ab": ["Ab", "Bb", "C", "Db", "Eb", "F", "G"],
    "A": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "Bb": ["Bb", "C", "D", "Eb", "F", "G", "A"],
    "B": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
}

_CHORD_TO_ROMAN_NUMERAL_MAJOR = {
    1: ("I", "major"),
    2: ("ii", "minor"),
    3: ("iii", "minor"),
    4: ("IV", "major"),
    5: ("V", "major"),
    6: ("vi", "minor"),
    7: ("vii dim", "diminished"),
}

# idk if this is working
_CHORD_TO_ROMAN_NUMERAL_MINOR = {
    1: ("i", "minor"),
    2: ("ii dim", "diminished"),
    3: ("III", "major"),
    4: ("iv", "minor"),
    5: ("v", "minor"),
    6: ("VI", "major"),
    7: ("VII", "major"),
}

def get_all_keys() -> List[Tuple[int, str]]:
    return list(PITCH_CLASS_TO_NOTE_NAME_SHARP.items())


def get_progression_midi_notes(
    key_pitch_class: int, mode: str, progression_degrees: Tuple[int, ...]
):
    scale = get_scale(key_pitch_class, mode)
    midi_notes = []
    for chord in progression_degrees:
        chord_midi_notes = get_chord(
            scale,
            mode=mode,
            root=chord,
            inversion=None,
            # triads only for now
            chord_type=5,
            extensions=[],
            borrowed=None,
        )[1]
        midi_notes.append(chord_midi_notes)
    return midi_notes


def get_progression_midi(
    midi_notes, num_plays_per_chord: int = 1, play_duration_in_beats: int = 2
):
    progression = []
    prev_beat = 0
    for chord_midi in midi_notes:
        for _ in range(num_plays_per_chord):
            progression.append(
                (
                    prev_beat,
                    prev_beat + play_duration_in_beats,
                    (chord_midi, None, None),
                )
            )
            prev_beat += play_duration_in_beats

    return progression


def get_progression_by_root_pitch_class(
    root_pitch_class: int, mode: str, chord_degrees: Tuple[int]
):
    midi_notes = get_progression_midi_notes(root_pitch_class, mode, chord_degrees)
    return get_progression_midi(midi_notes)


def get_row_iterator(
    progressions: Tuple[Tuple[str, Tuple[int, ...]], ...],
    keys: Iterable[Tuple[int, str]],
    instruments: List[Dict[str, Any]],
) -> Iterator[DatasetRowDescription]:
    # check that all chord progressions are unique
    assert len(progressions) == len(set(progressions))

    idx = 0
    for root_note_pitch_class, note_name in keys:
        for progression in progressions:
            for instrument_info in instruments:
                yield (
                    idx,
                    {
                        "instrument_info": instrument_info,
                        "progression": progression,
                        "note_name": note_name,
                        "root_note_pitch_class": root_note_pitch_class,
                    },
                )
                idx += 1


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    progression = row_info["progression"]
    note_name = row_info["note_name"]
    root_note_pitch_class = row_info["root_note_pitch_class"]

    # get soundfont information
    instrument_info = row_info["instrument_info"]
    midi_program_num = instrument_info["program"]
    midi_program_name = instrument_info["name"]
    midi_category = instrument_info["category"]

    mode, chord_degrees = progression
    chord_progression_label = f"{mode}-{chord_degrees}"
    chord_deg_str = "-".join(map(str, chord_degrees))
    progression_name = f"{mode}_{chord_deg_str}"
    cleaned_name = midi_program_name.replace(" ", "_")
    midi_file_path = (
        dataset_path
        / f"{note_name}_{progression_name}_{midi_program_num}_{cleaned_name}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{note_name}_{progression_name}_{midi_program_num}_{cleaned_name}.wav"
    )
    chord_midi = get_progression_by_root_pitch_class(
        root_note_pitch_class, mode, chord_degrees
    )
    midi_file = create_midi_file()
    midi_track = create_midi_track(
        # this information doesn't change the sound
        bpm=120,
        time_signature=(4, 4),
        key_root=note_name,
        track_name=midi_program_name,
        program=midi_program_num,
        channel=2,
    )
    write_progression(chord_midi, midi_track, channel=2)
    midi_file.tracks.append(midi_track)
    midi_file.save(midi_file_path)
    produce_synth_wav_from_midi(midi_file_path, synth_file_path)
    is_silent = is_wave_silent(synth_file_path)

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "key_note_name": note_name,
                "key_note_pitch_class": root_note_pitch_class,
                "chord_progression": chord_progression_label,
                "midi_program_num": midi_program_num,
                "midi_program_name": midi_program_name,
                "midi_category": midi_category,
                "midi_file_path": str(midi_file_path.relative_to(dataset_path)),
                "synth_file_path": str(synth_file_path.relative_to(dataset_path)),
                # e.g. TimGM6mb.sf2
                "synth_soundfont": DEFAULT_SOUNDFONT_LOCATION.parts[-1],
                "is_silent": is_silent,
            },
        )
    ]

def get_base_text_prompts(
    progression: Tuple[Tuple[str, Tuple[int, ...]], ...],
    note_name: str
) -> Iterator[str]:
    prompts = []

    mode, chord_degrees = progression
    chord_deg_str = "-".join(map(str, chord_degrees))

    if mode == "ionian":
        key_quality = "major"

        roman_numerals = tuple(_CHORD_TO_ROMAN_NUMERAL_MAJOR[chord_degree][0] for chord_degree in chord_degrees)

        # find chord roots
        if note_name in _NOTE_TO_MAJOR_SCALE:
            chord_roots = tuple(f"{_NOTE_TO_MAJOR_SCALE[note_name][chord_degree-1]} {_CHORD_TO_ROMAN_NUMERAL_MAJOR[chord_degree][1]}" for chord_degree in chord_degrees)
            chord_root_str = "-".join(map(str, chord_roots))
            prompts.append(f"The chords {chord_root_str}")

            chord_root_str = chord_root_str.replace(" major", "maj").replace(" minor", "min").replace(" diminished", "dim")
            prompts.append(f"The chords {chord_root_str}")
    else:
        key_quality = "minor"

        roman_numerals = tuple(_CHORD_TO_ROMAN_NUMERAL_MINOR[chord_degree][0] for chord_degree in chord_degrees)

        # find chord roots
        if note_name in _NOTE_TO_RELATIVE_MAJOR:
            note_name_relative = _NOTE_TO_RELATIVE_MAJOR[note_name]
            chord_roots = tuple(f"{_NOTE_TO_MAJOR_SCALE[note_name_relative][(chord_degree-3)%7]} {_CHORD_TO_ROMAN_NUMERAL_MINOR[chord_degree][1]}" for chord_degree in chord_degrees)
            chord_root_str = "-".join(map(str, chord_roots))
            prompts.append(f"The chords {chord_root_str}")

            chord_root_str = chord_root_str.replace(" major", "maj").replace(" minor", "min").replace(" diminished", "dim")
            prompts.append(f"The chords {chord_root_str}")

    roman_num_str = "-".join(map(str, roman_numerals))

    prompts.append(f"Progression {chord_deg_str} in {note_name} {key_quality}")
    prompts.append(f"Chord progression {chord_deg_str} in {note_name} {mode}")
    prompts.append(f"Chord progression {roman_num_str} in {note_name} {key_quality}")
    prompts.append(f"Chord progression {roman_num_str} in {note_name} {mode}")

    return prompts

def get_all_text_prompts(
    progression: Tuple[Tuple[str, Tuple[int, ...]], ...],
    note_name: str
) -> Iterator[str]:
    prompts = get_base_text_prompts(progression, note_name)

    if note_name in NOTE_NAME_TO_ENHARMONIC:
        new_prompts = get_base_text_prompts(progression, NOTE_NAME_TO_ENHARMONIC[note_name])
        prompts.extend(new_prompts)

    return prompts

def get_prompt_row_iterator(
    progressions: Tuple[Tuple[str, Tuple[int, ...]], ...],
    keys: Iterable[Tuple[int, str]]
) -> Iterator[DatasetRowDescription]:
    # check that all chord progressions are unique
    assert len(progressions) == len(set(progressions))

    idx = 0
    for root_note_pitch_class, note_name in keys:
        for progression in progressions:
            prompts = get_all_text_prompts(progression, note_name)

            for prompt in prompts:
                yield (
                    idx,
                    {
                        "progression": progression,
                        "note_name": note_name,
                        "root_note_pitch_class": root_note_pitch_class,
                        "prompt": prompt,
                    },
                )
                idx += 1


def prompt_row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    progression = row_info["progression"]
    note_name = row_info["note_name"]
    root_note_pitch_class = row_info["root_note_pitch_class"]
    prompt = row_info["prompt"]

    mode, chord_degrees = progression
    chord_progression_label = f"{mode}-{chord_degrees}"
   
    # record this row in the csv
    return [
        (
            row_idx,
            {
                "key_note_name": note_name,
                "key_note_pitch_class": root_note_pitch_class,
                "chord_progression": chord_progression_label,
                "prompt": prompt,
            },
        )
    ]

if __name__ == "__main__":
    """This requires 29.61 GB on disk.
    
    There are 20,976 files in the default configuration.
    """
    # configure the dataset
    dataset_name = "chord_progressions"
    # dataset_writer = DatasetWriter(
    #     dataset_name=dataset_name,
    #     save_to_parent_directory=OUTPUT_DIR,
    #     row_iterator=get_row_iterator(
    #         progressions=PROGRESSIONS,
    #         keys=get_all_keys(),
    #         instruments=get_instruments(
    #             ignore_atonal=True,
    #             ignore_polyphonic=True,
    #             ignore_highly_articulate=True,
    #             take_only_first_category=False,
    #         ),
    #     ),
    #     row_processor=row_processor,
    #     max_processes=8,
    # )

    # # create the dataset
    # dataset_df = dataset_writer.create_dataset()

    # # warn of any silent samples
    # num_silent_samples = dataset_df[dataset_df["is_silent"] == True].shape[0]  # noqa
    # if num_silent_samples > 0:
    #     warnings.warn(
    #         f"In the dataset, there were {num_silent_samples} silent samples.",
    #         UserWarning,
    #     )


    prompt_dataset_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_prompt_row_iterator(
            progressions=PROGRESSIONS,
            keys=get_all_keys(),
        ),
        row_processor=prompt_row_processor,
        max_processes=8,
        is_prompts=True,
    )

    # create the dataset
    dataset_df = prompt_dataset_writer.create_dataset()
