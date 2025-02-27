import warnings
from pathlib import Path
from typing import Tuple, List, Iterator, Dict, Any, Iterable
from itertools import product
from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.constants import MODES, PITCH_CLASS_TO_NOTE_NAME_SHARP, NOTE_NAME_TO_ENHARMONIC, NOTE_NAME_TO_PITCH_CLASS
from dataset.music.transforms import get_scale, get_tonic_midi_note_value
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_melody,
)
from dataset.synthetic.midi_instrument import get_instruments
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent

_PLAY_STYLE = {
    0: "UP",
    1: "DOWN",
}

_MODE_TO_SCALE_ALTERATION = {
    "ionian": ["no alterations", "sharp 3, sharp 6, and sharp 7"],
    "dorian": ["flat 3 and flat 7", "sharp 6"],
    "phrygian": ["flat 2, flat 3, flat 6, and flat 7", "flat 2"],
    "lydian": ["sharp 4", "sharp 3, sharp 4, flat 6, and flat 7"],
    "mixolydian": ["flat 7", "sharp 3 and sharp 6"],
    "aeolian": ["flat 3, flat 6, and flat 7", "no alterations"],
    "locrian": ["flat 2, flat 3, flat 5, flat 6, and flat 7", "flat 2 and flat 5"]
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

_MODE_TO_DOWN_INTERVAL = {
    "ionian": 0,
    "dorian": 2,
    "phrygian": 4,
    "lydian": 5,
    "mixolydian": 7,
    "aeolian": 9,
    "locrian": 11,
}

_MODE_TO_INDEX = {
    "ionian": 0,
    "dorian": 1,
    "phrygian": 2,
    "lydian": 3,
    "mixolydian": 4,
    "aeolian": 5,
    "locrian": 6,
}

def get_scale_midi(
    root_note_name: str,
    scale_mode: str,
    play_style: int,
    include_octave_above: bool = True,
):
    pitch_classes = get_scale(root_note_name, scale_mode)
    midi_tonic_val = get_tonic_midi_note_value(pitch_classes[0])
    offsets = MODES[scale_mode]
    if include_octave_above:
        offsets += (12,)

    notes = []
    for i in range(len(offsets)):
        notes.append(midi_tonic_val + offsets[i])

    if play_style == 1:
        # go down instead
        notes.reverse()

    # add timing for MIDI write
    timed_notes = []
    time_per_note = 1
    prev_beat = 0
    for n in notes:
        start_beat = prev_beat
        end_beat = start_beat + time_per_note
        timed_notes.append((start_beat, end_beat, (n, None)))
        prev_beat = end_beat

    return timed_notes


def get_all_scales(
    for_modes: Tuple[str] = (
        "ionian",
        "dorian",
        "phrygian",
        "lydian",
        "mixolydian",
        "aeolian",
        "locrian",
    ),
) -> List[Tuple[str, str]]:
    note_names = list(PITCH_CLASS_TO_NOTE_NAME_SHARP.values())
    return list(product(note_names, for_modes))


def get_row_iterator(
    scales: Iterable[Tuple[str, str]], instruments: List[Dict[str, Any]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for root_note, mode in scales:
        for play_style, play_style_name in _PLAY_STYLE.items():
            for instrument_info in instruments:
                yield (
                    idx,
                    {
                        "instrument_info": instrument_info,
                        "play_style": play_style,
                        "play_style_name": play_style_name,
                        "root_note": root_note,
                        "mode": mode,
                    },
                )
                idx += 1


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    play_style = row_info["play_style"]
    play_style_name = row_info["play_style_name"]
    root_note = row_info["root_note"]
    mode = row_info["mode"]
    # get soundfont information
    instrument_info = row_info["instrument_info"]
    midi_program_num = instrument_info["program"]
    midi_program_name = instrument_info["name"]
    midi_category = instrument_info["category"]

    cleaned_name = midi_program_name.replace(" ", "_")
    midi_file_path = (
        dataset_path
        / f"{root_note}_{mode}_{play_style_name}_{midi_program_num}_{cleaned_name}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{root_note}_{mode}_{play_style_name}_{midi_program_num}_{cleaned_name}.wav"
    )

    scale_midi = get_scale_midi(root_note, mode, play_style)
    midi_file = create_midi_file()
    midi_track = create_midi_track(
        # this information doesn't change the sound
        bpm=120,
        time_signature=(4, 4),
        key_root=root_note,
        track_name=midi_program_name,
        program=midi_program_num,
        channel=2,
    )
    write_melody(scale_midi, midi_track, channel=2)
    midi_file.tracks.append(midi_track)
    midi_file.save(midi_file_path)
    produce_synth_wav_from_midi(midi_file_path, synth_file_path)
    is_silent = is_wave_silent(synth_file_path)

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "root_note_name": root_note,
                "mode": mode,
                "play_style": play_style,
                "play_style_name": play_style_name,
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

def get_base_text_prompts(root_note: str, mode: str) -> Iterator[str]:
    prompts = [f"{root_note} {mode} scale"]
    prompts.append(f"{mode.capitalize()} scale starting on {root_note}")
    
    alterations = _MODE_TO_SCALE_ALTERATION[mode]
    prompts.append(f"{root_note} major scale with {alterations[0]}")
    prompts.append(f"{root_note} minor scale with {alterations[1]}")

    prompts.append(f"{mode.capitalize()} mode on {root_note}")

    # generate the notes in the scale
    # C dorian -> go down a whole step to find scale with C as its second note
    # step 1: get pitch class, go down 2, find 2 enharmonic scales (A# and Bb)
    # step 2: check if second note of that scale is root note (C)
    # step 3: starting at root note C, print out all the notes in that scale

    pitch_class = NOTE_NAME_TO_PITCH_CLASS[root_note]
    major_scale_root = PITCH_CLASS_TO_NOTE_NAME_SHARP[(pitch_class - _MODE_TO_DOWN_INTERVAL[mode]) % 12]

    # if the new root is not in the list of scales, try the enharmonic
    if major_scale_root not in _NOTE_TO_MAJOR_SCALE:
        if major_scale_root not in NOTE_NAME_TO_ENHARMONIC:
            return prompts
                
        major_scale_root = NOTE_NAME_TO_ENHARMONIC[major_scale_root]
    
    # if original root note is not in new scale, choose its enharmonic
    if root_note not in _NOTE_TO_MAJOR_SCALE[major_scale_root]:
        if major_scale_root not in NOTE_NAME_TO_ENHARMONIC:
            return prompts
                
        major_scale_root = NOTE_NAME_TO_ENHARMONIC[major_scale_root]

        # if new enharmonic scale does not exist, just return existing prompts
        if major_scale_root not in _NOTE_TO_MAJOR_SCALE:
            return prompts
    
    scale = _NOTE_TO_MAJOR_SCALE[major_scale_root]

    # get index of the root note in the scale
    index = scale.index(root_note)

    prompts.append(f"{major_scale_root} scale starting at {root_note}")

    notes_prompt = "The scale going up with notes "

    for i in range(7):
        notes_prompt += f"{scale[(index+i)%7]} "
    
    # add last note
    notes_prompt += f"{root_note}"

    prompts.append(notes_prompt)

    notes_prompt = "The scale going down with notes "
    for i in range(7,0,-1):
        notes_prompt += f"{scale[(index+i)%7]} "

    # add last note
    notes_prompt += f"{root_note}"

    prompts.append(notes_prompt)

    return prompts

def get_all_text_prompts(root_note: str, mode: str) -> Iterator[str]:
    prompts = get_base_text_prompts(root_note, mode)

    if root_note in NOTE_NAME_TO_ENHARMONIC:
        enharmonic = NOTE_NAME_TO_ENHARMONIC[root_note]
        new_prompts = get_base_text_prompts(enharmonic, mode)
        prompts.extend(new_prompts)

    return prompts

def get_prompt_row_iterator(
    scales: Iterable[Tuple[str, str]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for root_note, mode in scales:
        prompts = get_all_text_prompts(root_note, mode)
        for prompt in prompts:
            yield (
                idx,
                {
                    "root_note": root_note,
                    "mode": mode,
                    "prompt": prompt,
                },
            )
            idx += 1

def prompt_row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    root_note = row_info["root_note"]
    mode = row_info["mode"]
    prompt = row_info["prompt"]

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "root_note": root_note,
                "mode": mode,
                "prompt": prompt,
            },
        )
    ]


if __name__ == "__main__":
    """Requires 21.82 GB of disk space.
    
    This has 15,456 samples.
    """
    # configure the dataset
    dataset_name = "scales"
    # dataset_writer = DatasetWriter(
    #     dataset_name=dataset_name,
    #     save_to_parent_directory=OUTPUT_DIR,
    #     row_iterator=get_row_iterator(
    #         scales=get_all_scales(),
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

    prompts_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_prompt_row_iterator(
            scales=get_all_scales()
        ),
        row_processor=prompt_row_processor,
        max_processes=8,
        is_prompts=True,
    )

    dataset_df = prompts_writer.create_dataset()
