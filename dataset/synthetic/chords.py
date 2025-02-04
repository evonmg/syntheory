import string
import warnings
from pathlib import Path
from typing import Optional, List, Iterator, Iterable, Dict, Any, Tuple
from itertools import product
import random

# changed
import csv

from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.constants import (
    PITCH_CLASS_TO_NOTE_NAME_SHARP,
    PITCH_CLASS_TO_NOTE_NAME_ENHARMONIC,
)
from dataset.music.transforms import (
    get_major_triad,
    get_minor_triad,
    get_augmented_triad,
    get_diminished_triad,
)
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_progression,
)
from dataset.synthetic.midi_instrument import get_instruments
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription

_CHORD_MAP = {
    "major": get_major_triad,
    "minor": get_minor_triad,
    "aug": get_augmented_triad,
    "dim": get_diminished_triad,
}


def get_chord_midi(
    root_note_pitch_class: int,
    kind: str,
    inversion: Optional[str],
    num_plays: int = 4,
    play_duration_in_beats: int = 2,
):
    f = _CHORD_MAP[kind]
    pitch_classes, midi_notes, chord_name, _, _ = f(root_note_pitch_class, inversion)
    progression = []
    prev_beat = 0
    for _ in range(num_plays):
        progression.append(
            (prev_beat, prev_beat + play_duration_in_beats, (midi_notes, None, None))
        )
        prev_beat += play_duration_in_beats
    return progression


def get_note_name_from_pitch_class(pitch_class: int) -> str:
    return PITCH_CLASS_TO_NOTE_NAME_SHARP[pitch_class]


def get_all_chords() -> List[Tuple[int, str]]:
    note_names = list(PITCH_CLASS_TO_NOTE_NAME_SHARP.keys())
    chord_types = list(_CHORD_MAP.keys())
    return list(product(note_names, chord_types))

# Adds all of the chord configurations into a DatasetRowDescription
def get_row_iterator(
    chords: Iterable[Tuple[int, str]], instruments: List[Dict[str, Any]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for root_note_pitch_class, chord_type in chords:
        note_name = get_note_name_from_pitch_class(root_note_pitch_class)
        for inversion in [None, "6", "64"]:
            for instrument_info in instruments:
                yield (
                    idx,
                    {
                        "instrument_info": instrument_info,
                        "inversion": inversion,
                        "note_name": note_name,
                        "root_note_pitch_class": root_note_pitch_class,
                        "chord_type": chord_type,
                    },
                )
                idx += 1

# Creates a midi file for each row
def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    # get soundfont information
    note_name = row_info["note_name"]
    inversion = row_info["inversion"]
    root_note_pitch_class = row_info["root_note_pitch_class"]
    chord_type = row_info["chord_type"]
    instrument_info = row_info["instrument_info"]
    midi_program_num = instrument_info["program"]
    midi_program_name = instrument_info["name"]
    midi_category = instrument_info["category"]

    cleaned_name = midi_program_name.replace(" ", "_")
    midi_file_path = (
        dataset_path
        / f"{note_name}_{chord_type}_{inversion or '5'}_{midi_program_num}_{cleaned_name}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{note_name}_{chord_type}_{inversion or '5'}_{midi_program_num}_{cleaned_name}.wav"
    )
    # add chord text descriptions. except this time it makes one for every instrument which is definitely a waste of computation
    # text_file_path = (
    #     dataset_path
    #     / f"{note_name}_{chord_type}_{inversion or '5'}.csv"
    # )

    chord_midi = get_chord_midi(root_note_pitch_class, chord_type, inversion)
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

    # # create rows of text prompts
    # # examples of text prompts for chords:
    # # C minor, Cmin, Cm - note_name + chord_type + inversion
    # prompts = [f"{note_name} {chord_type} {inversion or '5'}", f"{note_name}{chord_type[:3]}{inversion or '5'}"]
    # if chord_type == "minor":
    #     prompts.append(f"{note_name}m{inversion or '5'}")
    # elif chord_type == "major":
    #     prompts.append(f"{note_name}M{inversion or '5'}")

    # # create csv file
    # with open(text_file_path, "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(prompts)

    is_silent = is_wave_silent(synth_file_path)

    return [
        (
            row_idx,
            {
                "root_note_name": note_name,
                "chord_type": chord_type,
                "inversion": inversion or "5",
                "root_note_is_accidental": note_name.endswith("#"),
                "root_note_pitch_class": root_note_pitch_class,
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

# Adds all of the chord configurations into a DatasetRowDescription
def get_row_iterator(
    chords: Iterable[Tuple[int, str]], instruments: List[Dict[str, Any]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for root_note_pitch_class, chord_type in chords:
        note_name = get_note_name_from_pitch_class(root_note_pitch_class)
        for inversion in [None, "6", "64"]:
            for instrument_info in instruments:
                yield (
                    idx,
                    {
                        "instrument_info": instrument_info,
                        "inversion": inversion,
                        "note_name": note_name,
                        "root_note_pitch_class": root_note_pitch_class,
                        "chord_type": chord_type,
                    },
                )
                idx += 1

# generates the actual prompts given the info
def generate_text_prompts(note_name: str, chord_type: str, inversion: str | None):
    prompts = []

    prompts.append(f"{note_name} {chord_type}{f" {inversion}" if inversion is not None else ''}")
    prompts.append(f"Generate a {note_name} {chord_type}{f" {inversion}" if inversion is not None else ''} chord")
    prompts.append(f"The chord {note_name} {chord_type}{f" {inversion}" if inversion is not None else ''}")
    prompts.append(f"Produce a {chord_type}{f" {inversion}" if inversion is not None else ''} chord with root {note_name}")
    prompts.append(f"Output the triad {note_name} {chord_type}{f" {inversion}" if inversion is not None else ''}")
    prompts.append(f"Invoke a{f" {inversion}" if inversion is not None else ''} triad that is {chord_type} with {note_name} as the tonic")
    prompts.append(f"Build a {chord_type}{f" {inversion}" if inversion is not None else ''} chord anchored on {note_name}")
    prompts.append(f"Express a{f" {inversion}" if inversion is not None else ''} {chord_type} chord with {note_name} as the root")
    prompts.append(f"Perform a{f" {inversion}" if inversion is not None else ''} chord rooted at {note_name} with quality {chord_type}")

    if inversion is None:
        prompts.append(f"Root position {note_name} {chord_type} chord")
        prompts.append(f"Sound a {note_name} {chord_type} triad in root position")
        prompts.append(f"The chord {note_name} {chord_type} in root position")
        prompts.append(f"Play a {chord_type} chord rooted at {note_name} in root position")
        prompts.append(f"Compose a {chord_type} root position chord with {note_name} as the root")
        prompts.append(f"A root position {chord_type} triad with bass {note_name}")
    else:
        prompts.append(f"{note_name} {chord_type} in the {inversion} inversion")
        if inversion == "6":
            prompts.append(f"1st inversion {note_name} {chord_type} chord")
            prompts.append(f"Sound a {note_name} {chord_type} triad in 1st inversion")
            prompts.append(f"The chord {note_name} {chord_type} in 1st inversion")
            prompts.append(f"Play a {chord_type} chord rooted at {note_name} in the 1st inversion")
            prompts.append(f"Compose a {chord_type} 1st inversion chord with {note_name} as the root")
            # A major 6 -> C# as the bass? how do I do this computationally lol
            # prompts.append(f"A root position {chord_type} triad with bass {note_name}")
        elif inversion == "64":
            prompts.append(f"2nd inversion {note_name} {chord_type} chord")
            prompts.append(f"Sound a {note_name} {chord_type} triad in 2nd inversion")
            prompts.append(f"The chord {note_name} {chord_type} in 2nd inversion")
            prompts.append(f"Play a {chord_type} chord rooted at {note_name} in the 2nd inversion")
            prompts.append(f"Compose a {chord_type} 2nd inversion chord with {note_name} as the root")

    return prompts

def get_all_text_prompts(note_name: str, chord_type: str, inversion: str | None):
    prompts = []
    prompts_added = generate_text_prompts(note_name, chord_type, inversion)
    prompts.extend(prompts_added)

    if chord_type == "dim":
        prompts_added = generate_text_prompts(note_name, "diminished", inversion)
        prompts.extend(prompts_added)
        prompts.append(f"{note_name}dim{f"{inversion}" if inversion is not None else ''}")
    elif chord_type == "aug":
        prompts_added = generate_text_prompts(note_name, "augmented", inversion)
        prompts.extend(prompts_added)
        prompts.append(f"{note_name}aug{f"{inversion}" if inversion is not None else ''}")
    elif chord_type == "major":
        prompts_added = generate_text_prompts(note_name, "maj", inversion)
        prompts.extend(prompts_added)
        prompts.append(f"{note_name}maj{f"{inversion}" if inversion is not None else ''}")
    elif chord_type == "minor":
        prompts_added = generate_text_prompts(note_name, "min", inversion)
        prompts.extend(prompts_added)
        prompts.append(f"{note_name}m{f"{inversion}" if inversion is not None else ''}")
        prompts.append(f"{note_name}min{f" {inversion}" if inversion is not None else ''}")

    if note_name[-1] == "#":
        sharp_note = note_name[0]
        prompts_added = generate_text_prompts(f"{sharp_note}-sharp", chord_type, inversion)
        prompts.extend(prompts_added)

        if chord_type == "dim":
            prompts_added = generate_text_prompts(f"{sharp_note}-sharp", "diminished", inversion)
            prompts.extend(prompts_added)
        elif chord_type == "aug":
            prompts_added = generate_text_prompts(f"{sharp_note}-sharp", "augmented", inversion)
            prompts.extend(prompts_added)
        
        letters = string.ascii_uppercase
        index = letters.index(sharp_note)
        flat_note = letters[(index + 1) % 7]
        prompts_added = generate_text_prompts(f"{flat_note}-flat", chord_type, inversion)
        prompts.extend(prompts_added)

        if chord_type == "dim":
            prompts_added = generate_text_prompts(f"{flat_note}-flat", "diminished", inversion)
            prompts.extend(prompts_added)
        elif chord_type == "aug":
            prompts_added = generate_text_prompts(f"{flat_note}-flat", "augmented", inversion)
            prompts.extend(prompts_added)
    else:
        prompts_added = generate_text_prompts(f"{note_name}-natural", chord_type, inversion)
        prompts.extend(prompts_added)

        if chord_type == "dim":
            prompts_added = generate_text_prompts(f"{note_name}-natural", "diminished", inversion)
            prompts.extend(prompts_added)
        elif chord_type == "aug":
            prompts_added = generate_text_prompts(f"{note_name}-natural", "augmented", inversion)
            prompts.extend(prompts_added)

    return prompts

# randomly samples labels that are not the current labels
def get_counterfactual_labels(note_name, chord_type, inversion) -> tuple[str, str, str]:
    note_names = list(PITCH_CLASS_TO_NOTE_NAME_SHARP.keys())
    chord_types = list(_CHORD_MAP.keys())
    inversions = [None, "6", "64"]

    rand_note_name = get_note_name_from_pitch_class(note_names[random.randint(0, len(note_names)-1)])
    while rand_note_name == note_name:
        rand_note_name = get_note_name_from_pitch_class(note_names[random.randint(0, len(note_names)-1)])

    rand_chord_type = chord_types[random.randint(0, len(chord_types)-1)]
    while rand_chord_type == chord_type:
        rand_chord_type = chord_types[random.randint(0, len(chord_types)-1)]

    rand_inversion = inversions[random.randint(0, len(inversions)-1)]
    while rand_inversion == inversion:
        rand_inversion = inversions[random.randint(0, len(inversions)-1)]

    return (rand_note_name, rand_chord_type, rand_inversion)

def get_prompt_row_iterator(
    chords: Iterable[Tuple[int, str]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for root_note_pitch_class, chord_type in chords:
        note_name = get_note_name_from_pitch_class(root_note_pitch_class)

        for inversion in [None, "6", "64"]:
            prompts = get_all_text_prompts(note_name, chord_type, inversion)

            for prompt in prompts:
                cf_note_name, cf_chord_type, cf_inversion = get_counterfactual_labels(note_name, chord_type, inversion)
                yield (
                    idx,
                    {
                        "inversion": inversion,
                        "note_name": note_name,
                        "root_note_pitch_class": root_note_pitch_class,
                        "chord_type": chord_type,
                        "prompt": prompt,
                        "cf_inversion": cf_inversion,
                        "cf_note_name": cf_note_name,
                        "cf_chord_type": cf_chord_type
                    },
                )
                idx += 1

def prompt_row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    note_name = row_info["note_name"]
    inversion = row_info["inversion"]
    root_note_pitch_class = row_info["root_note_pitch_class"]
    chord_type = row_info["chord_type"]
    prompt = row_info["prompt"]
    cf_inversion = row_info["cf_inversion"]
    cf_note_name = row_info["cf_note_name"]
    cf_chord_type = row_info["cf_chord_type"]

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "root_note_name": note_name,
                "chord_type": chord_type,
                "inversion": inversion or "5",
                "cf_root_note_name": cf_note_name,
                "cf_chord_type": cf_chord_type,
                "cf_inversion": cf_inversion,
                "root_note_pitch_class": root_note_pitch_class,
                "prompt": prompt
            },
        )
    ]

if __name__ == "__main__":
    """Requires 18.7 GB of disk space.
    
    There are 13,248 samples in the default configuration.
    """
    # configure the dataset
    dataset_name = "chords"
    # dataset_writer = DatasetWriter(
    #     dataset_name=dataset_name,
    #     save_to_parent_directory=OUTPUT_DIR,
    #     row_iterator=get_row_iterator(
    #         chords=get_all_chords(),
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

    # create prompts dataset
    prompts_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_prompt_row_iterator(
            get_all_chords()
        ),
        row_processor=prompt_row_processor,
        max_processes=8,
        is_prompts=True
    )

    prompts_writer.create_dataset()
