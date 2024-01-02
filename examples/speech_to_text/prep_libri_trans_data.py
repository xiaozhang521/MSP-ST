#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
import string
import csv

import numpy as np
import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
from torch.utils.data import Dataset
from tqdm import tqdm


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class ST_Dataset(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    def __init__(self, root: str, src_lang, tgt_lang: str, split: str, speed_perturb: bool = False, tokenizer: bool = False) -> None:
        #_root = Path(root) / f"{src_lang}-{tgt_lang}" / split
        _root = Path(root) / split
        wav_root, txt_root = _root / "audiofiles", _root 
        if tokenizer:
            txt_root = _root / "txt.tok"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir(), (_root, wav_root, txt_root)
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        #with open(txt_root / f"{split}.yaml") as f:
        #    segments = yaml.load(f, Loader=yaml.BaseLoader)
        segments=[]
        with open(txt_root / f"alignments.meta",encoding="utf-8") as f:
             lines=f.read().strip().split("\n")[1:]
             for line in lines:
                  segments.append({'duration':line.split("\t")[5],'wav':line.split("\t")[4]+".wav",'speaker_id':line.split("\t")[0]+"-"+line.split("\t")[1]})
        self.speed_perturb = [0.9, 1.0, 1.1] if speed_perturb and split.startswith("train") else None
        # Load source and target utterances
        for _lang in [src_lang, tgt_lang]:
            with open(txt_root / f"{split}.{_lang}",encoding="utf-8") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            try:
                sample_rate = torchaudio.info(wav_path.as_posix())[0].rate
            except TypeError:
                sample_rate = torchaudio.info(wav_path.as_posix()).sample_rate
            #seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                #offset = int(float(segment["offset"]) * sample_rate)
                offset=0
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{split}_{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment[src_lang],
                        segment[tgt_lang],
                        segment["speaker_id"] if "speaker_id" in segment else "spk1",
                        _id,
                    )
                )

    def __getitem__(self, n: int):
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]

        items = []
        if self.speed_perturb is None:
            waveform, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=n_frames)
            items.append([waveform, sr, n_frames, src_utt, tgt_utt, spk_id, utt_id])
        else:
            for speed in self.speed_perturb:
                sp_utt_id = f"sp{speed}_" + utt_id
                sp_n_frames = n_frames / speed
                if speed == 1.0:
                    waveform, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=n_frames)
                else:
                    waveform, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=n_frames)
                    effects = [
                        ["speed", f"{speed}"],
                        ["rate", f"{sr}"]
                    ]
                    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)

                items.append([waveform, sr, sp_n_frames, src_utt, tgt_utt, spk_id, sp_utt_id])
        return items

    def get_wav(self, n: int, speed_perturb=1.0):
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]

        if self.speed_perturb is None or speed_perturb == 1.0:
            waveform, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=n_frames)
        else:
            waveform, _ = torchaudio.load(wav_path, frame_offset=offset, num_frames=n_frames)
            effects = [
                ["speed", f"{speed_perturb}"],
                ["rate", f"{sr}"]
            ]
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
        return waveform

    def get_fast(self, n: int):
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]

        items = []
        if self.speed_perturb is None:
            items.append([wav_path, sr, n_frames, src_utt, tgt_utt, spk_id, utt_id])
        else:
            for speed in self.speed_perturb:
                sp_utt_id = f"sp{speed}_" + utt_id
                sp_n_frames = n_frames / speed
                items.append([wav_path, sr, sp_n_frames, src_utt, tgt_utt, spk_id, sp_utt_id])
        return items

    def get_src_text(self):
        src_text = []
        for item in self.data:
            src_text.append(item[4])
        return src_text

    def get_tgt_text(self):
        tgt_text = []
        for item in self.data:
            tgt_text.append(item[5])
        return tgt_text

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    splits = args.splits.split(",")
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    #cur_root = root / f"{src_lang}-{tgt_lang}"
    cur_root = root 
    if not cur_root.is_dir():
        print(f"{cur_root.as_posix()} does not exist. Skipped.")
        return
    if args.output_root is None:
        output_root = cur_root
    else:
        #output_root = Path(args.output_root).absolute() / f"{src_lang}-{tgt_lang}"
        output_root = Path(args.output_root).absolute()

    # Extract features
    if args.speed_perturb:
        zip_path = output_root / "fbank80_sp.zip"
    else:
        zip_path = output_root / "fbank80.zip"
    index = 0

    gen_feature_flag = False
    if not Path.exists(zip_path):
        gen_feature_flag = True

    if args.overwrite or gen_feature_flag:
        if args.speed_perturb:
            feature_root = output_root / "fbank80_sp"
        else:
            feature_root = output_root / "fbank80"
        feature_root.mkdir(exist_ok=True)

        for split in splits:
            print(f"Fetching split {split}...")
            dataset = ST_Dataset(root.as_posix(), src_lang, tgt_lang, split, args.speed_perturb, args.tokenizer)
            is_train_split = split.startswith("train")
            print("Extracting log mel filter bank features...")
            if is_train_split and args.cmvn_type == "global":
                print("And estimating cepstral mean and variance stats...")
                gcmvn_feature_list = []

            for idx in tqdm(range(len(dataset))):
                items = dataset.get_fast(idx)
                for item in items:
                    index += 1
                    wav_path, sr, _, _, _, _, utt_id = item

                    features_path = (feature_root / f"{utt_id}.npy").as_posix()
                    if not os.path.exists(features_path):
                        sp = 1.0
                        if dataset.speed_perturb is not None:
                            sp = float(utt_id.split("_")[0].replace("sp", ""))
                        waveform = dataset.get_wav(idx, sp)
                        if waveform.shape[1] == 0:
                            continue
                        features = extract_fbank_features(waveform, sr, Path(features_path))

                        if split == 'train' and args.cmvn_type == "global" and not utt_id.startswith("sp"):
                            if len(gcmvn_feature_list) < args.gcmvn_max_num:
                                gcmvn_feature_list.append(features)

                if is_train_split and args.size != -1 and index > args.size:
                    break

            if is_train_split and args.cmvn_type == "global":
                # Estimate and save cmv
                stats = cal_gcmvn_stats(gcmvn_feature_list)
                with open(output_root / "gcmvn.npz", "wb",encoding="utf-8") as f:
                    np.savez(f, mean=stats["mean"], std=stats["std"])

        # Pack features into ZIP
        print("ZIPing features...")
        create_zip(feature_root, zip_path)

        # Clean up
        shutil.rmtree(feature_root)

    gen_manifest_flag = False
    for split in splits:
        if not Path.exists(output_root / f"{split}_{args.task}.tsv"):
            gen_manifest_flag = True
            break

    train_text = []
    if args.overwrite or gen_manifest_flag:
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(zip_path)
        # Generate TSV manifest
        print("Generating manifest...")
        for split in splits:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            if args.task == "st" and args.add_src:
                manifest["src_text"] = []
            if args.gtranslation and is_train_split and args.task == "st":
                with open(root / f"{split}/train_gtranslate.{tgt_lang}",encoding="utf-8") as f:
                    gtranslates=f.read().strip().split("\n")

            dataset = ST_Dataset(args.data_root, src_lang, tgt_lang, split, args.speed_perturb, args.tokenizer)
            for idx in range(len(dataset)):
                items = dataset.get_fast(idx)
                for item in items:
                    _, sr, n_frames, src_utt, tgt_utt, speaker_id, utt_id = item
                    manifest["id"].append(utt_id)
                    manifest["audio"].append(zip_manifest[utt_id])
                    duration_ms = int(n_frames / sr * 1000)
                    manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
                    if args.lowercase_src:
                        src_utt = src_utt.lower()
                    if args.rm_punc_src:
                        for w in string.punctuation:
                            src_utt = src_utt.replace(w, "")
                    manifest["tgt_text"].append(src_utt if args.task == "asr" else tgt_utt)
                    if args.task == "st" and args.add_src:
                        manifest["src_text"].append(src_utt)
                    manifest["speaker"].append(speaker_id)
                    
                    if args.gtranslation and is_train_split and args.task == "st":
                        manifest["id"].append(manifest["id"][-1])
                        manifest["audio"].append(manifest["audio"][-1])
                        manifest["n_frames"].append(manifest["n_frames"][-1])
                        manifest["tgt_text"].append(gtranslates[idx])
                        if args.task == "st" and args.add_src:
                            manifest["src_text"].append(manifest["src_text"][-1])
                        manifest["speaker"].append(manifest["speaker"][-1])
                         
                if is_train_split and args.size != -1 and len(manifest["id"]) > args.size:
                    break
            if is_train_split:
                if args.task == "st" and args.add_src and args.share:
                    train_text.extend(manifest["src_text"])
                train_text.extend(manifest["tgt_text"])

            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split)
            save_df_to_tsv(df, output_root / f"{split}_{args.task}.tsv")

    # Generate vocab
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
    asr_spm_filename = None
    gen_vocab_flag = True

    if args.task == "st" and args.add_src:
        if args.share:
            if args.st_spm_prefix is not None:
                gen_vocab_flag = False
                spm_filename_prefix = args.st_spm_prefix
            else:
                spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}_share"
            asr_spm_filename = spm_filename_prefix + ".model"
        else:
            if args.st_spm_prefix is not None:
                gen_vocab_flag = False
                spm_filename_prefix = args.st_spm_prefix
            assert args.asr_prefix is not None
            asr_spm_filename = args.asr_prefix + ".model"
    elif args.task == "asr":
        if args.asr_prefix is not None:
            gen_vocab_flag = False
            spm_filename_prefix = args.asr_prefix

    if gen_vocab_flag:
        if len(train_text) == 0:
            print("Loading the training text to build dictionary...")

            for split in args.SPLITS:
                if split.startswith("train"):
                    csv_path = output_root / f"{split}_{args.task}.tsv"
                    with open(csv_path,encoding="utf-8") as f:
                        reader = csv.DictReader(
                            f,
                            delimiter="\t",
                            quotechar=None,
                            doublequote=False,
                            lineterminator="\n",
                            quoting=csv.QUOTE_NONE,
                        )

                    if args.task == "st" and args.add_src and args.share:
                        for e in reader:
                            src_utt = dict(e)["src_text"]
                            if args.lowercase_src:
                                src_utt = src_utt.lower()
                            if args.rm_punc_src:
                                for w in string.punctuation:
                                    src_utt = src_utt.replace(w, "")
                                src_utt = src_utt.replace("  ", "")
                            train_text.append(src_utt)
                    tgt_text = [dict(e)["tgt_text"] for e in reader]
                    train_text.extend(tgt_text)

        with NamedTemporaryFile(mode="w",encoding="utf-8") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                output_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )

    # Generate config YAML
    yaml_filename = f"config_{args.task}.yaml"
    if args.task == "st" and args.add_src and args.share:
        yaml_filename = f"config_{args.task}_share.yaml"

    gen_config_yaml(
        output_root,
        spm_filename_prefix + ".model",
        yaml_filename=yaml_filename,
        specaugment_policy=args.specaugment_policy,
        cmvn_type=args.cmvn_type,
        gcmvn_path=(
            output_root / "gcmvn.npz" if args.cmvn_type == "global"
            else None
        ),
        asr_spm_filename=asr_spm_filename,
        share_src_and_tgt=True if args.task == "asr" else False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--output-root", "-o", default=None, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, default="st", choices=["asr", "st"])
    parser.add_argument("--src-lang", type=str, required=True, help="source language")
    parser.add_argument("--tgt-lang", type=str, required=True, help="target language")
    parser.add_argument("--splits", type=str, default="train,dev,test", help="dataset splits")
    parser.add_argument("--size", default=-1, type=int)
    parser.add_argument("--speed-perturb", action="store_true", default=False,
                        help="apply speed perturbation on wave file")
    parser.add_argument("--share", action="store_true",
                        help="share the tokenizer and dictionary of the transcription and translation")
    parser.add_argument("--add-src", action="store_true", help="add the src text for st task")
    parser.add_argument("--asr-prefix", type=str, default=None, help="prefix of the asr dict")
    parser.add_argument("--st-spm-prefix", type=str, default=None, help="prefix of the existing st dict")
    parser.add_argument("--lowercase-src", action="store_true", help="lowercase the source text")
    parser.add_argument("--rm-punc-src", action="store_true", help="remove the punctuation of the source text")
    parser.add_argument("--tokenizer", action="store_true", help="use tokenizer txt")
    parser.add_argument("--specaugment-policy", default="lb", help="specaugment policy, if disable set none")
    parser.add_argument("--cmvn-type", default="utterance",
                        choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the existing files")
    parser.add_argument("--gtranslation", default=True, action="store_true", help="use gtranslaiton")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help=(
                            "Maximum number of sentences to use to estimate"
                            "global mean and variance"
                            ))
    args = parser.parse_args()
    args.size

    process(args)


if __name__ == "__main__":
    main()
