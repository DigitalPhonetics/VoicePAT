# This code is based on
# https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/voxceleb_prepare.py
import csv
import logging
import random
from pathlib import Path
import sys  # noqa F401
import numpy as np
import torch
import torchaudio
import soundfile as sf
from tqdm.contrib import tqdm
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_libri_prepare.pkl"
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
ENROL_CSV = "enrol.csv"
SAMPLERATE = 16000

def prepare_libri(
    data_folder,
    save_folder,
    splits=["train", "dev"],
    split_ratio=[90, 10],
    seg_dur=3.0,
    amp_th=5e-04,
    num_utt=None,
    num_spk=None,
    random_segment=False,
    skip_prep=False,
    anon = False,
    utt_selected_ways="spk-random",
):
    """
    Prepares the csv files for the libri datasets.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original libri  dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    verification_pairs_file : str
        txt file containing the verification split.
    splits : list
        List of splits to prepare from ['train', 'dev']
    split_ratio : list
        List if int for train and validation splits
    seg_dur : int
        Segment duration of a chunk in seconds (e.g., 3.0 seconds).
    amp_th : float
        removes segments whose average amplitude is below the
        given threshold.
    source : str
        Path to the folder where the VoxCeleb dataset source is stored.
    num_utt: float
        How many utterances for each speaker used for training
    num_spk: float
        How many speakers used for training
    random_segment : bool
        Train random segments
    skip_prep: Bool
        If True, skip preparation.

    Example
    -------
    >>> from libri_prepare import prepare_libri
    >>> data_folder = 'LibriSpeech/train-clean-360/'
    >>> save_folder = 'libri/'
    >>> splits = ['train', 'dev']
    >>> split_ratio = [90, 10]
    >>> prepare_voxceleb(data_folder, save_folder, splits, split_ratio)
    """

    if skip_prep:
        return
    # Create configuration for easily skipping data_preparation stage
    conf = {
        "data_folder": data_folder,
        "splits": splits,
        "split_ratio": split_ratio,
        "save_folder": save_folder,
        "seg_dur": seg_dur,
        "num_utt": num_utt,
        "num_spk": num_spk,
    }

    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True, parents=True)

    # Setting ouput files
    save_opt = save_folder / OPT_FILE
    save_csv_train = save_folder / TRAIN_CSV
    save_csv_dev = save_folder / DEV_CSV


    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return

    # Additional checks to make sure the data folder contains VoxCeleb data
    if "," in data_folder:
        data_folder = [Path(dir) for dir in data_folder.replace(" ", "").split(",")]
    else:
        data_folder = [Path(data_folder)]

    # _check_voxceleb1_folders(data_folder, splits)

    msg = "\tCreating csv file for the Libri Dataset.."
    logger.info(msg)

    # Split data into 90% train and 10% validation (verification split)
    wav_lst_train, wav_lst_dev = _get_utt_split_lists(
        data_folder, split_ratio, num_utt, num_spk, anon, utt_selected_ways
    )

    # Creating csv file for training data
    if "train" in splits:
        prepare_csv(
            seg_dur, wav_lst_train, save_csv_train, random_segment, amp_th
        )

    if "dev" in splits:
        prepare_csv(seg_dur, wav_lst_dev, save_csv_dev, random_segment, amp_th)


    # Saving options (useful to skip this phase when already done)
    save_pkl(conf, str(save_opt))


def skip(splits, save_folder, conf):
    """
    Detects if the voxceleb data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking csv files
    skip = True

    split_files = {
        "train": TRAIN_CSV,
        "dev": DEV_CSV,
    }
    for split in splits:
        if not Path(save_folder, split_files[split]).is_file():
            skip = False
    #  Checking saved options
    save_opt = save_folder / OPT_FILE
    if skip is True:
        if save_opt.is_file():
            opts_old = load_pkl(str(save_opt))
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip

# Used for verification split
def _get_utt_split_lists(
    data_folders, split_ratio, num_utt='ALL', num_spk='ALL', anon=False, utt_selected_ways="spk-random"
):
    """
    Tot. number of speakers libri-360=921
    Splits the audio file list into train and dev.
    """
    train_lst = []
    dev_lst = []

    print("Getting file list...")
    for data_folder in data_folders:
        if anon:
            suffix = 'wav'
        else:
            suffix = 'flac'
        # if anon:
        #     path = os.path.join(data_folder, "*.wav")
        # else:
        #     path = os.path.join(data_folder, "LibriSpeech/train-clean-360""**", "**", "*.flac")
            # avoid test speakers for train and dev splits
        spk_files = {}
        spks_pure = []
        full_utt = 0
        for f in data_folder.glob(f'**/*.{suffix}'):
            # temp = f.split("/")[-1].split(".")[0]
            temp = f.stem
            used_id = "-".join(temp.split('-')[-3:-1])
            spk_id = temp.split('-')[0]
            if spk_id not in spks_pure:
                spks_pure.append(spk_id)
                
            if used_id not in spk_files:
                spk_files[used_id] = []
            if f not in spk_files[used_id]:
                full_utt += 1
                spk_files[used_id].append(str(f.absolute()))
        
        selected_list = []
        selected_spk = {}
        #select the number of speakers
        if num_spk != 'ALL':
            print("selected %s speakers for training"%num_spk)
            selected_spks_pure = random.sample(spks_pure,int(num_spk))
            for k,v in spk_files.items():
                if k.split('-')[0] in selected_spks_pure:
                    selected_spk[k] = v
            #selected_spk = dict(random.sample(spk_files.items(), int(num_spk)))
        elif num_spk == 'ALL':
            print("selected all speakers for training")
            selected_spk = spk_files
        else:
            sys.exit("invalid $utt_spk value")

            # select the number of utterances for each speaker-sess-id
        if num_utt != 'ALL':
            # select the number of utterances for each speaker-sess-id
            if utt_selected_ways == 'spk-sess':
                print("selected %s utterances for each selected speaker-sess-id" % num_utt)
                for spk in selected_spk:
                    if len(selected_spk[spk]) >= int(num_utt):
                        selected_list.extend(random.sample(selected_spk[spk], int(num_utt)))
                    else:
                        selected_list.extend(selected_spk[spk])

            elif utt_selected_ways == 'spk-random':
                print("randomly selected %s utterances for each selected speaker-id" % num_utt)
                selected_spks_pure = {}
                for k, v in selected_spk.items():
                    spk_pure = k.split('-')[0]
                    if spk_pure not in selected_spks_pure:
                        selected_spks_pure[spk_pure] = []
                    selected_spks_pure[spk_pure].extend(v)

                selected_spk = selected_spks_pure

                for spk in selected_spk:
                    if len(selected_spk[spk]) >= int(num_utt):
                        selected_list.extend(random.sample(selected_spk[spk], int(num_utt)))
                    else:
                        selected_list.extend(selected_spk[spk])

            elif utt_selected_ways == 'spk-diverse-sess':
                print("diversely selected %s utterances for each selected speaker-id" % num_utt)
                selected_spks_pure = {}
                for k, v in selected_spk.items():
                    spk_pure = k.split('-')[0]
                    if spk_pure not in selected_spks_pure:
                        selected_spks_pure[spk_pure] = []
                    selected_spks_pure[spk_pure].append(v)

                selected_spk = selected_spks_pure

                for spk in selected_spk:
                    num_each_sess = round(num_utt / len(selected_spk[spk]))  # rounded up
                    for utts in selected_spk[spk]:
                        if len(utts) >= int(num_each_sess):
                            selected_list.extend(random.sample(utts, int(num_each_sess)))
                        else:
                            selected_list.extend(selected_spk[spk])


        elif num_utt == 'ALL':
            print("selected all utterances for each selected speaker")

            for value in selected_spk.values():
                for v in value:
                    selected_list.append(v)

        else:
            sys.exit("invalid $utt_num value")

        flat = []
        for row in selected_list:
            if 'list' in str(type(row)):
                for x in row:
                    if x not in flat:
                        flat.append(x)
            else:
                if row not in flat:
                    flat.append(row)

        selected_list = flat
        random.shuffle(selected_list)
        
        full = f'Full training set:{full_utt}'
        used = f'Used for training:{len(selected_list)}'
        print(full)
        print(used)

        split = int(0.01 * split_ratio[0] * len(selected_list))
        train_snts = selected_list[:split]
        dev_snts = selected_list[split:]

        train_lst.extend(train_snts)
        dev_lst.extend(dev_snts)

    return train_lst, dev_lst


def _get_chunks(seg_dur, audio_id, audio_duration):
    """
    Returns list of chunks
    """
    num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

    chunk_lst = [
        audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
        for i in range(num_chunks)
    ]

    return chunk_lst


def prepare_csv(seg_dur, wav_lst, csv_file, random_segment=False, amp_th=0):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    random_segment: bool
        Read random segments
    amp_th: float
        Threshold on the average amplitude on the chunk.
        If under this threshold, the chunk is discarded.

    Returns
    -------
    None
    """

    msg = f'\t"Creating csv lists in  {csv_file}..."'
    logger.info(msg)

    csv_output = [["ID", "duration", "wav", "start", "stop", "spk_id"]]

    # For assigning unique ID to each chunk
    my_sep = "--"
    entry = []
    problematic_wavs = []
    # Processing all the wav files in the list
    for wav_file in tqdm(wav_lst, dynamic_ncols=True):
        # Getting sentence and speaker ids
        try:
            temp = wav_file.split("/")[-1].split(".")[0]
            [spk_id, sess_id, utt_id] = temp.split('-')[-3:]
        except ValueError:
            logger.info(f"Malformed path: {wav_file}")
            continue
        audio_id = my_sep.join([spk_id, sess_id, utt_id.split(".")[0]])

        # Reading the signal (to retrieve duration in seconds)
        try:
            audio_duration = sf.info(wav_file).duration
            #signal, fs = torchaudio.load(wav_file)
        except RuntimeError:
            problematic_wavs.append(wav_file)
            continue
        #signal = signal.squeeze(0)

        if random_segment:
            #audio_duration = signal.shape[0] / SAMPLERATE
            start_sample = 0
            #stop_sample = signal.shape[0]
            stop_sample = int(audio_duration * SAMPLERATE)

            # Composition of the csv_line
            csv_line = [
                audio_id,
                str(audio_duration),
                wav_file,
                start_sample,
                stop_sample,
                spk_id,
            ]
            entry.append(csv_line)
        else:
            #audio_duration = signal.shape[0] / SAMPLERATE
            signal, fs = torchaudio.load(wav_file)
            signal = signal.squeeze(0)

            uniq_chunks_list = _get_chunks(seg_dur, audio_id, audio_duration)
            for chunk in uniq_chunks_list:
                s, e = chunk.split("_")[-2:]
                start_sample = int(float(s) * SAMPLERATE)
                end_sample = int(float(e) * SAMPLERATE)

                #  Avoid chunks with very small energy
                mean_sig = torch.mean(np.abs(signal[start_sample:end_sample]))
                if mean_sig < amp_th:
                    continue

                # Composition of the csv_line
                csv_line = [
                    chunk,
                    str(audio_duration),
                    wav_file,
                    start_sample,
                    end_sample,
                    spk_id,
                ]
                entry.append(csv_line)

    print(f'Skipped {len(problematic_wavs)} invalid audios')
    csv_output = csv_output + entry

    # Writing the csv lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        for line in csv_output:
            csv_writer.writerow(line)

    # Final prints
    msg = f"\t{csv_file} successfully created!"
