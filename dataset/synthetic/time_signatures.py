import warnings
from pathlib import Path
from typing import Tuple, List, Iterable, Iterator, Optional
from config import OUTPUT_DIR, DEFAULT_SOUNDFONT_LOCATION
from dataset.music.midi import ClickTrackConfig
from dataset.audio.synth import produce_synth_wav_from_midi
from dataset.audio.wav import is_wave_silent, random_trim, trim
from dataset.music.track import create_click_track_midi
from dataset.music.midi import is_compound_time_signature
from dataset.synthetic.metronome_configs import CLICK_CONFIGS
from dataset.synthetic.dataset_writer import DatasetWriter, DatasetRowDescription

_NUMBERS_TO_WORDS = {
    2: "two",
    3: "three",
    4: "four",
    6: "six",
    8: "eight",
    9: "nine",
    12: "twelve",
}

_NUMBERS_TO_NOTE_LENGTH = {
    2: "half",
    4: "quarter",
    8: "eighth",
}

_NUMBERS_TO_NOTE_LENGTH_BRITISH = {
    2: "minim",
    4: "crotchet",
    8: "quaver",
}

def get_all_time_signatures() -> List[Tuple[int, int]]:
    return [(2, 2), (2, 4), (3, 4), (3, 8), (4, 4), (6, 8), (9, 8), (12, 8)]


def get_row_iterator(
    time_signatures: Iterable[Tuple[int, int]],
    click_configs: Iterable[ClickTrackConfig],
    num_reverb_levels: int,
    num_random_offsets: int,
    target_duration_per_sample_in_sec: float,
    bpm: int = 120,
    seed: Optional[int] = None,
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for time_signature in time_signatures:
        for reverb_level in range(num_reverb_levels):
            for config in click_configs:
                yield (
                    idx,
                    {
                        "bpm": bpm,
                        "time_signature": time_signature,
                        "click_config": config,
                        "reverb_level": reverb_level,
                        "num_random_offsets": num_random_offsets,
                        "target_duration_per_sample_in_sec": target_duration_per_sample_in_sec,
                        "seed": seed,
                    },
                )
                idx += num_random_offsets


def row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    config = row_info["click_config"]
    time_signature = row_info["time_signature"]
    reverb_level = row_info["reverb_level"]
    bpm = row_info["bpm"]
    num_random_offsets = row_info["num_random_offsets"]
    target_duration_per_sample_in_sec = row_info["target_duration_per_sample_in_sec"]
    midi_program_num = config.midi_program_num

    # play approx 30 seconds of audio
    total_beats_to_play = int((time_signature[1] / 4) * (bpm // 2))
    midi_file = create_click_track_midi(
        bpm,
        # ~ approximately 20 seconds but not exactly, we over-shoot it
        # so that we have space to move around after the wav is created
        total_beats_to_play,
        midi_file=None,
        time_signature=time_signature,
        config=config,
        reverb_level=reverb_level,
    )

    time_signature_readable_name = str(time_signature)[1:-1].replace(", ", "_")
    midi_file_path = (
        dataset_path
        / f"{time_signature_readable_name}_{bpm}_bpm_{config.name}_reverb_level_{reverb_level}.mid"
    )
    synth_file_path = (
        dataset_path
        / f"{time_signature_readable_name}_{bpm}_bpm_{config.name}_reverb_level_{reverb_level}.wav"
    )
    midi_file.save(midi_file_path)

    # play the MIDI, realizing it to a waveform
    produce_synth_wav_from_midi(midi_file_path, synth_file_path, show_logs=True)

    # force each sample to be 30 seconds
    trim(synth_file_path, synth_file_path, target_duration=30.0, overwrite_output=True)

    rows = []
    for i in range(num_random_offsets):
        # produce random trim from this sample
        offset_path = synth_file_path.parent / (
            synth_file_path.stem + f"_offset_{i}.wav"
        )
        offset_time = random_trim(
            synth_file_path,
            offset_path,
            target_duration=target_duration_per_sample_in_sec,
            overwrite_output=False,
            seed=row_info["seed"],
        )
        is_silent = is_wave_silent(offset_path)

        # record this row in the csv
        rows.append(
            (
                row_idx + i,
                {
                    "time_signature": time_signature,
                    "time_signature_beats": time_signature[0],
                    "time_signature_subdivision": time_signature[1],
                    "is_compound": int(is_compound_time_signature(time_signature)),
                    "bpm": bpm,
                    "click_config_name": config.name,
                    "midi_program_num": midi_program_num,
                    "midi_file_path": str(midi_file_path.relative_to(dataset_path)),
                    "synth_file_path": str(synth_file_path.relative_to(dataset_path)),
                    "offset_file_path": str(offset_path.relative_to(dataset_path)),
                    "offset_time": str(offset_time),
                    # e.g. TimGM6mb.sf2
                    "synth_soundfont": DEFAULT_SOUNDFONT_LOCATION.parts[-1],
                    "reverb_level": reverb_level,
                    "is_silent": is_silent,
                },
            )
        )
    return rows

from typing import List, Tuple

def get_all_text_prompts(time_signature: Tuple[int, int]) -> List[str]:
    prompts = []
    num, denom = time_signature  # Unpacking for readability
    
    # Core time signature prompts
    prompts.append(f"{num}/{denom} time")
    prompts.append(f"{num}/{denom} meter")
    prompts.append(f"{num}/{denom} rhythm")
    prompts.append(f"{_NUMBERS_TO_WORDS[num]}-{_NUMBERS_TO_WORDS[denom]} time")
    prompts.append(f"Time signature: {num}/{denom}")
    prompts.append(f"Musical time signature of {num}/{denom}")
    prompts.append(f"{num} beats per measure, each a {_NUMBERS_TO_NOTE_LENGTH[denom]} note")
    prompts.append(f"{num} {_NUMBERS_TO_NOTE_LENGTH[denom]} notes in each measure")
    prompts.append(f"{num} beats per bar ({_NUMBERS_TO_NOTE_LENGTH[denom]} unit beat)")
    prompts.append(f"A steady {num}/{denom} groove")
    prompts.append(f"A melody in {num}/{denom} meter")
    prompts.append(f"Generate a song in {num}/{denom}")
    prompts.append(f"Music written in {num}/{denom} time")
    prompts.append(f"Rhythm structured in {num}/{denom} notation")
    prompts.append(f"The time signature is {num}/{denom}")
    prompts.append(f"The time signature is divided by {denom} with each measure containing {num} beats")
    prompts.append(f"The time signature consists of {num} beats per measure, with each beat being a {_NUMBERS_TO_NOTE_LENGTH[denom]}-note")
    prompts.append(f"In {num}/{denom} time, the measure is divided into {num} parts")
    prompts.append(f"In {num}/{denom} meter, the bar consists of {num} beats")
    prompts.append(f"The rhythm structure is defined by the time signature {num}/{denom}")
    prompts.append(f"This composition is set in a time signature of {num}/{denom}")
    prompts.append(f"This piece is in {num}/{denom} time, with {num} beats per measure")
    
    # Add more phrasing variations
    prompts.append(f"Time with {num} {_NUMBERS_TO_NOTE_LENGTH[denom]} notes per measure")
    prompts.append(f"Time with {num} {_NUMBERS_TO_NOTE_LENGTH_BRITISH[denom]}s per bar")
    prompts.append(f"Generate a song with {num}/{denom} time signature")
    prompts.append(f"Generate a song with {_NUMBERS_TO_WORDS[num]}-{_NUMBERS_TO_WORDS[denom]} meter")
    prompts.append(f"{num} beats per measure ({_NUMBERS_TO_NOTE_LENGTH[denom]} note beats)")
    prompts.append(f"Time signature given by {num} {_NUMBERS_TO_NOTE_LENGTH[denom]} notes")
    prompts.append(f"Time signature divided by {_NUMBERS_TO_NOTE_LENGTH[denom]} notes, {num} of them per measure")
    prompts.append(f"Musical meter: {num} {_NUMBERS_TO_NOTE_LENGTH[denom]} note beats per measure")
    prompts.append(f"Composition written in {num} over {denom} time")
    prompts.append(f"Notation style with {num} {_NUMBERS_TO_NOTE_LENGTH[denom]} note pulses per bar")
    prompts.append(f"A melody set in {num}-{denom} rhythm")
    prompts.append(f"A steady {num}/{denom} groove")
    prompts.append(f"A song with {num}/{denom} time signature")
    prompts.append(f"Generate a melody in {num}/{denom} time")
    prompts.append(f"Create a rhythm in {num}/{denom}")
    prompts.append(f"A rhythmic structure based on {num}/{denom}")
    prompts.append(f"Notation style: {num}/{denom} meter")
    prompts.append(f"Generate a groove in {num}/{denom} time")
    prompts.append(f"Song structure based on {num}/{denom} time signature")
    prompts.append(f"A rhythmic pulse in {num}/{denom}")
    prompts.append(f"A groove based on {num}/{denom} time signature")

    # Special cases
    if time_signature == (4, 4):
        prompts.append("Common time")
        prompts.append("4/4 meter, also known as common time")
        prompts.append("A song structured in common time")
        prompts.append("A steady 4/4 rhythm with a driving pulse")
        prompts.append("A standard beat pattern in 4/4 time")

    elif time_signature == (2, 2):
        prompts.append("Cut time")
        prompts.append("Cut time rhythm")
        prompts.append("A song structured in cut common time")
        prompts.append("A fast 2/2 feel with half-note pulses")

    # Compound meters (e.g., 6/8, 9/8, 12/8)
    elif num in {6, 9, 12} and denom in {8, 16}:
        beats_per_bar = num // 3
        note_length = denom // 2  # E.g., 8 → dotted quarter, 16 → dotted eighth
        prompts.append(f"{beats_per_bar} dotted {_NUMBERS_TO_NOTE_LENGTH[note_length]} beats per measure")
        prompts.append(f"A swinging feel with {beats_per_bar} beats per bar")
        prompts.append(f"{beats_per_bar}-beat feel (each beat = dotted {_NUMBERS_TO_NOTE_LENGTH[note_length]})")
        prompts.append(f"A shuffle rhythm in {num}/{denom}")
        prompts.append(f"A compound time groove in {num}/{denom}")
        prompts.append(f"A syncopated pattern in {num}/{denom} meter")
        prompts.append(f"A lilting {num}/{denom} feel, often heard in folk music")

    # Triple meter & waltz feel (3/4, 6/8)
    elif num == 3 and denom == 4:
        prompts.append(f"Waltz time with quarter note beats")
        prompts.append(f"Triple meter groove with quarter note subdivisions")
        prompts.append(f"A steady 3/4 feel, characteristic of waltzes")
        prompts.append(f"A triple time signature groove, emphasizing the first beat")
        prompts.append(f"A smooth swaying rhythm in 3/4 meter")

    if time_signature == (6, 8):
        prompts.append(f"Compound duple meter groove with dotted quarter beats")  
        prompts.append(f"A lilting rhythm with two beats per measure (each a dotted quarter note)")  
        prompts.append(f"A rhythmic pattern emphasizing two strong beats in 6/8 time")  
    elif time_signature == (9, 8):
        prompts.append(f"Compound triple meter groove with dotted quarter beats")  
        prompts.append(f"A rolling rhythm with three beats per measure (each a dotted quarter note)")  
        prompts.append(f"A triplet-based feel in 9/8 time, emphasizing three strong pulses")  
        prompts.append(f"A waltz-like rhythm with an extra lilt in 9/8 meter")  
        prompts.append(f"A rhythm pattern divided into three groups of three eighth notes per measure")
    elif time_signature == (12, 8):
        prompts.append(f"Compound quadruple meter groove with dotted quarter beats")  
        prompts.append(f"A rolling groove with four beats per measure (each a dotted quarter note)")  
        prompts.append(f"A rhythm with four strong pulses, each subdivided into triplets")  
        prompts.append(f"A dance-like rhythm in 12/8 time, giving a smooth swaying effect")  
    
    # Genre-specific prompts
    prompts.append(f"A jazz improvisation in {num}/{denom}")
    prompts.append(f"A blues shuffle in {num}/{denom}")
    prompts.append(f"A rock groove in {num}/{denom} time")
    prompts.append(f"A progressive metal riff in {num}/{denom}")
    prompts.append(f"A classical composition in {num}/{denom} meter")
    prompts.append(f"A traditional folk melody in {num}/{denom}")
    prompts.append(f"A Latin rhythm with {num}/{denom} meter")
    prompts.append(f"A syncopated funk groove in {num}/{denom}")
    prompts.append(f"A cinematic score in {num}/{denom}, evoking an epic feel")
    
    return prompts

def get_prompt_row_iterator(
    time_signatures: Iterable[Tuple[int, int]],
) -> Iterator[DatasetRowDescription]:
    idx = 0
    for time_signature in time_signatures:
        prompts = get_all_text_prompts(time_signature)
        for prompt in prompts:
            yield (
                idx,
                {
                    "time_signature": time_signature,
                    "prompt": prompt,
                },
            )
            idx += 1

def prompt_row_processor(
    dataset_path: Path, row: DatasetRowDescription
) -> List[DatasetRowDescription]:
    row_idx, row_info = row

    time_signature = row_info["time_signature"]
    prompt = row_info["prompt"]

    rows = []

    # record this row in the csv
    rows.append(
        (
            row_idx,
            {
                "time_signature": time_signature,
                "time_signature_beats": time_signature[0],
                "time_signature_subdivision": time_signature[1],
                "is_compound": int(is_compound_time_signature(time_signature)),
                "prompt": prompt,
            },
        )
    )
    return rows

if __name__ == "__main__":
    """This requires 1.48 GB.
    Contains 1,200 samples in the default configuration.
    """
    # configure the dataset
    dataset_name = "time_signatures"
    # dataset_writer = DatasetWriter(
    #     dataset_name=dataset_name,
    #     save_to_parent_directory=OUTPUT_DIR,
    #     row_iterator=get_row_iterator(
    #         get_all_time_signatures(),
    #         click_configs=CLICK_CONFIGS,
    #         num_reverb_levels=3,
    #         num_random_offsets=10,
    #         target_duration_per_sample_in_sec=4.0,
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
    #         f"In the {dataset_name} dataset, there were {num_silent_samples} silent samples.",
    #         UserWarning,
    #     )

    dataset_writer = DatasetWriter(
        dataset_name=dataset_name,
        save_to_parent_directory=OUTPUT_DIR,
        row_iterator=get_prompt_row_iterator(
            get_all_time_signatures(),
        ),
        row_processor=prompt_row_processor,
        max_processes=8,
        is_prompts=True,
    )

    dataset_df = dataset_writer.create_dataset()