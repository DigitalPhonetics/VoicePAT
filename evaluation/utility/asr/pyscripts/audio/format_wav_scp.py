#!/usr/bin/env python3
import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import humanfriendly
import kaldiio
import numpy as np
import resampy
import soundfile
from tqdm import tqdm
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.fileio.vad_scp import VADScpReader
from espnet.utils.cli_utils import get_commandline_args


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def str2int_tuple(integers: str) -> Optional[Tuple[int, ...]]:
    """

    >>> str2int_tuple('3,4,5')
    (3, 4, 5)

    """
    assert check_argument_types()
    if integers.strip() in ("none", "None", "NONE", "null", "Null", "NULL"):
        return None
    return tuple(map(int, integers.strip().split(",")))


def vad_trim(vad_reader: VADScpReader, uttid: str, wav: np.array, fs: int) -> np.array:
    # Conduct trim wtih vad information

    assert check_argument_types()
    assert uttid in vad_reader

    vad_info = vad_reader[uttid]
    total_length = sum(int((time[1] - time[0]) * fs) for time in vad_info)
    new_wav = np.zeros((total_length,), dtype=wav.dtype)
    start_frame = 0
    for time in vad_info:
        # Note: we regard vad as [xxx, yyy)
        duration = int((time[1] - time[0]) * fs)
        orig_start_frame = int(time[0] * fs)
        orig_end_frame = orig_start_frame + duration

        end_frame = start_frame + duration
        new_wav[start_frame:end_frame] = wav[orig_start_frame:orig_end_frame]

        start_frame = end_frame

    return new_wav


class SegmentsExtractor:
    """Emulating kaldi extract-segments.cc

    Args:
        segments (str): The file format is
            "<segment-id> <recording-id> <start-time> <end-time>\n"
            "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5\n"
    """

    def __init__(self, fname: str, segments: str = None):
        assert check_argument_types()
        self.wav_scp = fname
        self.wav_dict = {}
        with open(self.wav_scp, "r") as f:
            for line in f:
                recodeid, wavpath = line.strip().split(None, 1)
                if recodeid in self.wav_dict:
                    raise RuntimeError(f"{recodeid} is duplicated")
                self.wav_dict[recodeid] = wavpath

        self.segments = segments
        self.segments_dict = {}
        with open(self.segments, "r") as f:
            for line in f:
                sps = line.rstrip().split(None)
                if len(sps) != 4:
                    raise RuntimeError("Format is invalid: {}".format(line))
                uttid, recodeid, st, et = sps
                self.segments_dict[uttid] = (recodeid, float(st), float(et))

                if recodeid not in self.wav_dict:
                    raise RuntimeError(
                        'Not found "{}" in {}'.format(recodeid, self.wav_scp)
                    )

    def generator(self):
        recodeid_counter = {}
        for utt, (recodeid, st, et) in self.segments_dict.items():
            recodeid_counter[recodeid] = recodeid_counter.get(recodeid, 0) + 1

        cached = {}
        for utt, (recodeid, st, et) in self.segments_dict.items():
            if recodeid not in cached:
                wavpath = self.wav_dict[recodeid]

                if wavpath.endswith("|"):
                    # Streaming input e.g. cat a.wav |
                    with kaldiio.open_like_kaldi(wavpath, "rb") as f:
                        with BytesIO(f.read()) as g:
                            retval = soundfile.read(g)
                else:
                    retval = soundfile.read(wavpath)
                cached[recodeid] = retval
            retval = cached[recodeid]

            # Keep array until the last query
            recodeid_counter[recodeid] -= 1
            if recodeid_counter[recodeid] == 0:
                cached.pop(recodeid)

            yield utt, self._return(retval, st, et)

    def _return(self, array, st, et):
        if isinstance(array, (tuple, list)):
            array, rate = array

        # Convert starting time of the segment to corresponding sample number.
        # If end time is -1 then use the whole file starting from start time.
        if et != -1:
            return array[int(st * rate) : int(et * rate)], rate
        else:
            return array[int(st * rate) :], rate


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='Create waves list from "wav.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp")
    parser.add_argument("outdir")
    parser.add_argument(
        "--name",
        default="wav",
        help="Specify the prefix word of output file name " 'such as "wav.scp"',
    )
    parser.add_argument("--segments", default=None)
    parser.add_argument(
        "--fs",
        type=humanfriendly_or_none,
        default=None,
        help="If the sampling rate specified, " "Change the sampling rate.",
    )
    parser.add_argument("--audio-format", default="wav")
    parser.add_argument("--vad_based_trim", type=str, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ref-channels", default=None, type=str2int_tuple)
    group.add_argument("--utt2ref-channels", default=None, type=str)
    args = parser.parse_args()

    out_num_samples = Path(args.outdir) / f"utt2num_samples"

    if args.ref_channels is not None:

        def utt2ref_channels(x) -> Tuple[int, ...]:
            return args.ref_channels

    elif args.utt2ref_channels is not None:
        utt2ref_channels_dict = read_2column_text(args.utt2ref_channels)

        def utt2ref_channels(x, d=utt2ref_channels_dict) -> Tuple[int, ...]:
            chs_str = d[x]
            return tuple(map(int, chs_str.split()))

    else:
        utt2ref_channels = None

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_wavscp = Path(args.outdir) / f"{args.name}.scp"
    if args.segments is not None:
        if args.vad_based_trim is not None:
            raise NotImplementedError(
                "VAD based trim for data with segments is not supported"
            )
        # Note: kaldiio supports only wav-pcm-int16le file.
        extractor = SegmentsExtractor(args.scp, segments=args.segments)
        if args.audio_format.endswith("ark"):
            fark = open(Path(args.outdir) / f"data_{args.name}.ark", "wb")
            fscp = out_wavscp.open("w")
        else:
            writer = SoundScpWriter(
                args.outdir,
                out_wavscp,
                format=args.audio_format,
            )

        with out_num_samples.open("w") as fnum_samples:
            for uttid, (wave, rate) in tqdm(extractor.generator()):
                # wave: (Time,) or (Time, Nmic)
                if wave.ndim == 2 and utt2ref_channels is not None:
                    wave = wave[:, utt2ref_channels(uttid)]

                if args.fs is not None and args.fs != rate:
                    # FIXME(kamo): To use sox?
                    wave = resampy.resample(wave, rate, args.fs, axis=0)
                    rate = args.fs
                if args.audio_format.endswith("ark"):
                    if "flac" in args.audio_format:
                        suf = "flac"
                    elif "wav" in args.audio_format:
                        suf = "wav"
                    else:
                        raise RuntimeError("wav.ark or flac")

                    # NOTE(kamo): Using extended ark format style here.
                    # This format is incompatible with Kaldi
                    kaldiio.save_ark(
                        fark,
                        {uttid: (wave, rate)},
                        scp=fscp,
                        append=True,
                        write_function=f"soundfile_{suf}",
                    )

                else:
                    writer[uttid] = rate, wave
                fnum_samples.write(f"{uttid} {len(wave)}\n")
    else:
        if args.audio_format.endswith("ark"):
            fark = open(Path(args.outdir) / f"data_{args.name}.ark", "wb")
        else:
            wavdir = Path(args.outdir) / f"data_{args.name}"
            wavdir.mkdir(parents=True, exist_ok=True)

        do_vad = False
        if args.vad_based_trim is not None:
            vad_reader = VADScpReader(args.vad_based_trim)
            do_vad = True

        with Path(args.scp).open("r") as fscp, out_wavscp.open(
            "w"
        ) as fout, out_num_samples.open("w") as fnum_samples:
            for line in tqdm(fscp):
                uttid, wavpath = line.strip().split(None, 1)

                if wavpath.endswith("|"):
                    # Streaming input e.g. cat a.wav |
                    with kaldiio.open_like_kaldi(wavpath, "rb") as f:
                        with BytesIO(f.read()) as g:
                            wave, rate = soundfile.read(g)
                            if wave.ndim == 2 and utt2ref_channels is not None:
                                wave = wave[:, utt2ref_channels(uttid)]

                        if args.fs is not None and args.fs != rate:
                            # FIXME(kamo): To use sox?
                            wave = resampy.resample(wave, rate, args.fs, axis=0)
                            rate = args.fs

                        if do_vad:
                            logging.info("conduct triming from {}".format(wave.shape))
                            wave = vad_trim(vad_reader, uttid, wave, rate)
                            logging.info("conduct triming into {}".format(wave.shape))

                        if args.audio_format.endswith("ark"):
                            if "flac" in args.audio_format:
                                suf = "flac"
                            elif "wav" in args.audio_format:
                                suf = "wav"
                            else:
                                raise RuntimeError("wav.ark or flac")

                            # NOTE(kamo): Using extended ark format style here.
                            # This format is incompatible with Kaldi
                            kaldiio.save_ark(
                                fark,
                                {uttid: (wave, rate)},
                                scp=fout,
                                append=True,
                                write_function=f"soundfile_{suf}",
                            )
                        else:
                            owavpath = str(wavdir / f"{uttid}.{args.audio_format}")
                            soundfile.write(owavpath, wave, rate)
                            fout.write(f"{uttid} {owavpath}\n")
                else:
                    wave, rate = soundfile.read(wavpath)
                    if wave.ndim == 2 and utt2ref_channels is not None:
                        wave = wave[:, utt2ref_channels(uttid)]
                        save_asis = False

                    elif args.audio_format.endswith("ark"):
                        save_asis = False

                    elif do_vad:
                        save_asis = False

                    elif Path(wavpath).suffix == "." + args.audio_format and (
                        args.fs is None or args.fs == rate
                    ):
                        save_asis = True

                    else:
                        save_asis = False

                    if save_asis:
                        # Neither --segments nor --fs are specified and
                        # the line doesn't end with "|",
                        # i.e. not using unix-pipe,
                        # only in this case,
                        # just using the original file as is.
                        fout.write(f"{uttid} {wavpath}\n")
                    else:
                        if args.fs is not None and args.fs != rate:
                            # FIXME(kamo): To use sox?
                            wave = resampy.resample(wave, rate, args.fs, axis=0)
                            rate = args.fs

                        if do_vad:
                            wave = vad_trim(vad_reader, uttid, wave, rate)

                        if args.audio_format.endswith("ark"):
                            if "flac" in args.audio_format:
                                suf = "flac"
                            elif "wav" in args.audio_format:
                                suf = "wav"
                            else:
                                raise RuntimeError("wav.ark or flac")

                            # NOTE(kamo): Using extended ark format style here.
                            # This format is not supported in Kaldi.
                            kaldiio.save_ark(
                                fark,
                                {uttid: (wave, rate)},
                                scp=fout,
                                append=True,
                                write_function=f"soundfile_{suf}",
                            )
                        else:
                            owavpath = str(wavdir / f"{uttid}.{args.audio_format}")
                            soundfile.write(owavpath, wave, rate)
                            fout.write(f"{uttid} {owavpath}\n")
                fnum_samples.write(f"{uttid} {len(wave)}\n")


if __name__ == "__main__":
    main()