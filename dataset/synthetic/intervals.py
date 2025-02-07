import warnings
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any
from itertools import product
from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.constants import PITCH_CLASS_TO_NOTE_NAME_SHARP
from dataset.music.midi import (
    create_midi_file,
    create_midi_track,
    write_melody,
    write_progression,
)
from dataset.synthetic.midi_instrument import get_instruments
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription

_PLAY_STYLE = {
    0: "UP",
    1: "DOWN",
    2: "UNISON",
}

_INTERVALS = {
    1: "m2",
    2: "M2",
    3: "m3",
    4: "M3",
    5: "P4",
    6: "d5",
    7: "P5",
    8: "m6",
    9: "M6",
    10: "m7",
    11: "M7",
    12: "P8",
}

_SPELLED_NUMS = {
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
}

_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

_ENHARMONICS = {
    "A#": "Bb",
    "B": "Cb",
    "C": "B#",
    "C#": "Db",
    "D#": "Eb",
    "E": "Fb",
    "F": "E#",
    "F#": "Gb",
    "G#": "Ab"
}

def write_interval_midi(
    base_midi_note_val: int,
    interval_size: int,
    play_style: int,
    midi_track,
    channel: int,
):
    # all intervals are computed relative to the base note midi value, going UP
    midi_note_vals = (base_midi_note_val, base_midi_note_val + interval_size)

    notes = []
    if play_style == 0:
        # UP
        prev_beat = 0
        for _ in range(4):
            notes.append((prev_beat, prev_beat + 1, (midi_note_vals[0], None)))
            notes.append((prev_beat + 1, prev_beat + 2, (midi_note_vals[1], None)))
            prev_beat += 2
        return write_melody(notes, midi_track, channel=channel)
    elif play_style == 1:
        # DOWN
        prev_beat = 0
        for _ in range(4):
            notes.append((prev_beat, prev_beat + 1, (midi_note_vals[1], None)))
            notes.append((prev_beat + 1, prev_beat + 2, (midi_note_vals[0], None)))
            prev_beat += 2
        return write_melody(notes, midi_track, channel=channel)
    elif play_style == 2:
        # UNISON
        prev_beat = 0
        for _ in range(4 * 2):
            notes.append((prev_beat, prev_beat + 1, (midi_note_vals, None, None)))
            prev_beat += 1

        return write_progression(notes, midi_track, channel=channel)


def get_note_name_from_pitch_class(pitch_class: int) -> str:
    return PITCH_CLASS_TO_NOTE_NAME_SHARP[pitch_class]


def get_base_note_midi_note_values() -> Iterator[int]:
    return iter(range(48 + 12, 59 + 1 + 12))


def get_interval_values() -> Iterator[int]:
    # return 1 to 12 (m2, to P8)
    return iter(range(1, 12 + 1))


def get_all_interval_midi_settings() -> List[Tuple[int, int]]:
    return list(product(get_base_note_midi_note_values(), get_interval_values()))


def get_row_iterator(
    intervals: List[Tuple[int, int]], instruments: List[Dict[str, Any]]
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for midi_base_note, midi_interval_val in intervals:
        note_name = get_note_name_from_pitch_class(midi_base_note % 12)
        for play_style in _PLAY_STYLE.keys():
            for instrument_info in instruments:
                play_style_name = _PLAY_STYLE[play_style]
                yield (
                    idx,
                    {
                        "instrument_info": instrument_info,
                        "play_style_name": play_style_name,
                        "play_style": play_style,
                        "note_name": note_name,
                        "midi_interval_val": midi_interval_val,
                        "midi_base_note": midi_base_note,
                    },
                )
                idx += 1


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row
    note_name = row_info["note_name"]
    play_style = row_info["play_style"]
    play_style_name = row_info["play_style_name"]
    midi_interval_val = row_info["midi_interval_val"]
    midi_base_note = row_info["midi_base_note"]

    # get soundfont information
    instrument_info = row_info["instrument_info"]
    midi_program_num = instrument_info["program"]
    midi_program_name = instrument_info["name"]
    midi_category = instrument_info["category"]
    cleaned_name = midi_program_name.replace(" ", "_")

    midi_file_path = (
        dataset_path
        / f"{note_name}_{midi_interval_val}_{play_style_name}_{midi_program_num}_{cleaned_name}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{note_name}_{midi_interval_val}_{play_style_name}_{midi_program_num}_{cleaned_name}.wav"
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
    write_interval_midi(
        midi_base_note,
        midi_interval_val,
        play_style,
        midi_track,
        channel=2,
    )
    midi_file.tracks.append(midi_track)
    midi_file.save(midi_file_path)
    produce_synth_wav_from_midi(midi_file_path, synth_file_path)

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "root_note_name": note_name,
                "root_note_pitch_class": midi_base_note % 12,
                "interval": midi_interval_val,
                "play_style": play_style,
                "play_style_name": play_style_name,
                "midi_note_val": midi_base_note,
                "midi_program_num": midi_program_num,
                "midi_program_name": midi_program_name,
                "midi_category": midi_category,
                "midi_file_path": str(midi_file_path.relative_to(dataset_path)),
                "synth_file_path": str(synth_file_path.relative_to(dataset_path)),
                # e.g. TimGM6mb.sf2
                "synth_soundfont": DEFAULT_SOUNDFONT_LOCATION.parts[-1],
                "is_silent": is_wave_silent(synth_file_path),
            },
        )
    ]

def get_interval_notes(note_name: str, midi_interval_val: int) -> tuple[str]:
    # if note_name = A, interval_name = 2 (M2), return A+2 (B), A-2 (G)
    # index = 0
    index = _NOTES.index(note_name)

    # _NOTES[2] = B
    up_note = _NOTES[(index+midi_interval_val)%len(_NOTES)]
    # _NOTES[(-2)%12] = _NOTES[10] = G
    down_note = _NOTES[(index-midi_interval_val)%len(_NOTES)]

    # this does not always work if interval is minor - e.g. C + m3 should not be D# but Eb
    if midi_interval_val in [3,6,8,10] and up_note[-1] == "#":
        # if note name is natural key
        if note_name[-1] != "#":
            up_note = _ENHARMONICS[up_note]

    # also does not work for down notes if note_name = C or F
    # C - M2 = Bb, not A#
    if note_name == "C":
        if midi_interval_val in [2,4,9,11]:
            down_note = _ENHARMONICS[down_note]
    elif note_name == "F":
        if midi_interval_val in [2,4,7,9,11]:
            down_note = _ENHARMONICS[down_note]

    # careful with m2
    # C + m2 = Db, not C#
    if midi_interval_val == 1 and note_name not in ["E","B"]:
        if note_name[-1] != "#":
            up_note = _ENHARMONICS[down_note]

    # if up note is C and root note is not natural
    if note_name[-1] == "#":
        if up_note in ["C", "F"]:
            up_note = _ENHARMONICS[up_note]

        if down_note in ["C", "F"]:
            down_note = _ENHARMONICS[down_note]

    # change A# in certain cases
    if note_name[-1] != "#" and note_name not in ["B","E"]:
        if up_note == "A#":
            up_note = "Bb"
        
        if down_note == "A#":
            down_note = "Bb"

    # change down major 7th in certain cases
    if note_name[-1] != "#" and midi_interval_val == 11 and down_note[-1] == "#":
        down_note = _ENHARMONICS[down_note]

    return (up_note, down_note)

def get_all_text_prompts(midi_interval_val: int, note_name: str) -> List[str]:
    prompts = []

    interval_name = _INTERVALS[midi_interval_val]
    interval_nth_name = f"{interval_name[1]}th"
    if interval_name[-1] == "2":
        interval_nth_name = "2nd"
    elif interval_name[-1] == "3":
        interval_nth_name = "3rd"

    prompts.append(f"Generate the interval {interval_name} starting at {note_name}")
    prompts.append(f"{interval_name} starting at {note_name}")
    prompts.append(f"{note_name} going up a {interval_name}")
    prompts.append(f"{note_name} going down a {interval_name}")

    spelled_num = _SPELLED_NUMS[int(interval_name[-1])]

    if interval_name[0] == "m":
        prompts.append(f"A minor {interval_nth_name} starting at {note_name}")
        prompts.append(f"Minor interval of a {spelled_num} starting at {note_name}")
        prompts.append(f"min{interval_name[1]} interval from {note_name}")
    elif interval_name[0] == "M":
        prompts.append(f"A major {interval_nth_name} starting at {note_name}")
        prompts.append(f"Major interval of a {spelled_num} starting at {note_name}")
        prompts.append(f"maj{interval_name[1]} interval from {note_name}")
    elif interval_name == "d5":
        prompts.append(f"An augmented 4th starting at {note_name}")
        prompts.append(f"A diminished 5th starting at {note_name}")
        prompts.append(f"Diminished interval of a {spelled_num} starting at {note_name}")
        prompts.append(f"dim{interval_name[1]} interval from {note_name}")
    elif interval_name[0] == "P":
        prompts.append(f"A perfect {interval_nth_name} starting at {note_name}")
        prompts.append(f"Perfect interval of a {spelled_num} starting at {note_name}")
        prompts.append(f"aug{interval_name[1]} interval from {note_name}")
    
    if interval_name[1] == "8":
        prompts.append(f"A perfect octave starting at {note_name}")

    # TODO: start at note, then go up/down the specified interval - make function to calculate this somehow
    up_note, down_note = get_interval_notes(note_name, midi_interval_val)

    # change behavior for A# and D# specifically
    if note_name not in ["A#", "D#"]:
        # don't go up major 7ths if note name is a sharp note
        if not (midi_interval_val == 11 and note_name[-1] == "#"):
            prompts.append(f"The interval given by the notes {note_name} going up to a {up_note}")
            prompts.append(f"Start at note {note_name} and go up to a {up_note}")
        # don't go down minor 2nds if note name is a sharp note
        if not (midi_interval_val == 1 and note_name[-1] == "#"):
            prompts.append(f"The interval given by the notes {note_name} going down to a {down_note}")
            prompts.append(f"Start at note {note_name} and go down to a {down_note}")
    elif note_name in ["A#", "D#"]:
        # down note is a natural note - change A# to Bb
        if down_note[-1] != "#" and midi_interval_val != 1:
            prompts.append(f"The interval given by the notes {_ENHARMONICS[note_name]} going down to a {down_note}")
            prompts.append(f"Start at note {_ENHARMONICS[note_name]} and go down to a {down_note}")
        else:
            prompts.append(f"The interval given by the notes {note_name} going down to a {down_note}")
            prompts.append(f"Start at note {note_name} and go down to a {down_note}")
        if up_note[-1] != "#" and midi_interval_val != 11:
            prompts.append(f"The interval given by the notes {_ENHARMONICS[note_name]} going up to a {up_note}")
            prompts.append(f"Start at note {_ENHARMONICS[note_name]} and go up to a {up_note}")
        else:
            prompts.append(f"The interval given by the notes {note_name} going up to a {up_note}")
            prompts.append(f"Start at note {note_name} and go up to a {up_note}")

    return prompts

def get_prompt_row_iterator(
    intervals: List[Tuple[int, int]]) -> Iterator[DatasetRowDescription]:
    idx = 0
    for midi_base_note, midi_interval_val in intervals:
        note_name = get_note_name_from_pitch_class(midi_base_note % 12)

        prompts = get_all_text_prompts(midi_interval_val, note_name)
        for prompt in prompts:
            yield (
                idx,
                {
                    "note_name": note_name,
                    "midi_interval_val": midi_interval_val,
                    "midi_base_note": midi_base_note,
                    "prompt": prompt
                },
            )
            idx += 1

def prompt_row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row
    note_name = row_info["note_name"]
    midi_interval_val = row_info["midi_interval_val"]
    midi_base_note = row_info["midi_base_note"]
    prompt = row_info["prompt"]

    # record this row in the csv
    return [
        (
            row_idx,
            {
                "root_note_name": note_name,
                "root_note_pitch_class": midi_base_note % 12,
                "interval": midi_interval_val,
                "midi_note_val": midi_base_note,
                "prompt": prompt,
            },
        )
    ]


if __name__ == "__main__":
    """This takes ~56.1 GB of disk space.
    
    There are 39,744 samples in the default configuration.
    """
    # configure the dataset
    dataset_name = "intervals"
    # dataset_writer = DatasetWriter(
    #     dataset_name=dataset_name,
    #     save_to_parent_directory=OUTPUT_DIR,
    #     row_iterator=get_row_iterator(
    #         intervals=get_all_interval_midi_settings(),
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

    # # check the resulting info csv / dataframe
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
            intervals=get_all_interval_midi_settings()
        ),
        row_processor=prompt_row_processor,
        max_processes=8,
        is_prompts=True,
    )

    dataset_df = prompts_writer.create_dataset()