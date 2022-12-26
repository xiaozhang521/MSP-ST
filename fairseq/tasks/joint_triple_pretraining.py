# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace
from collections import OrderedDict
import contextlib
import torch

from fairseq.data import Dictionary, encoders, RoundRobinZipDatasets
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.data.audio.triple_dataset import TripleDatasetCreator,S2TTripleDataConfig,TripleDataset

from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask


logger = logging.getLogger(__name__)


@register_task("joint_triple_pretraining")
class JointTriplePretrainingTask(SpeechToTextTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )

    def __init__(self, args, tgt_dict, src_dict=None):
        super().__init__(args, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.all_tasks = ["st","mt","asr"]
        self.data_cfg = S2TTripleDataConfig(op.join(args.data, args.config_yaml))

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TTripleDataConfig(op.join(args.data, args.config_yaml))
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        src_dict = None
        if getattr(data_cfg, "share_src_and_tgt", False):
            asr_vocab_filename = data_cfg.vocab_filename
        else:
            asr_vocab_filename = getattr(data_cfg, "asr_vocab_filename", None)
        if asr_vocab_filename is not None:
            dict_path = op.join(args.data, asr_vocab_filename)
            if not op.isfile(dict_path):
                raise FileNotFoundError(f"Dict not found: {dict_path}")
            src_dict = Dictionary.load(dict_path)
            logger.info(
                f"asr dictionary size ({asr_vocab_filename}): " f"{len(src_dict):,}"
            )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict, src_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        is_valid_split = split.startswith("dev") or split.startswith("valid")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        if self.data_cfg.src_bpe_tokenizer is not None:
            src_bpe_tokenizer = self.build_src_bpe(self.args)
        else:
            src_bpe_tokenizer = bpe_tokenizer
            # if self.data_cfg.share_src_and_tgt:
            #     src_bpe_tokenizer = bpe_tokenizer
            # else:
            #     src_bpe_tokenizer = None
        is_decode=True
        if is_train_split:
            is_decode=False
            train_files = split.split(",")
            st_files = []
            mt_files = []
            asr_files = [] 
            for file_name in train_files:
                if "st" in file_name:
                    st_files.append(file_name)
                elif "mt" in file_name:
                    mt_files.append(file_name)
                elif "asr" in file_name:
                    asr_files.append(file_name)
                else:
                    raise ValueError(
                    'Please specify the file type, the file name should contain one of "st,mt,asr" tag'
                    )
            split_st = ','.join(st_files)
            split_mt = ','.join(mt_files)
            split_asr = ','.join(asr_files)

            data_dict=[]
            if len(mt_files) > 0:
                mt_data = TripleDatasetCreator.from_tsv(
                    self.args.data,
                    self.data_cfg,
                    split_mt,
                    self.tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    is_train_split=is_train_split,
                    epoch=epoch,
                    seed=self.args.seed,
                    src_dict=self.src_dict,
                    src_bpe_tokenizer=src_bpe_tokenizer,
                    data_type="mt"
                )
                data_dict.append(("mt",mt_data))

            if len(asr_files) > 0:
                asr_data = TripleDatasetCreator.from_tsv(
                    self.args.data,
                    self.data_cfg,
                    split_asr,
                    self.tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    is_train_split=is_train_split,
                    epoch=epoch,
                    seed=self.args.seed,
                    src_dict=self.src_dict,
                    src_bpe_tokenizer=src_bpe_tokenizer,
                    data_type="asr"
                )
                data_dict.append(("asr",asr_data))

            if len(st_files) > 0:
                st_data = TripleDatasetCreator.from_tsv(
                    self.args.data,
                    self.data_cfg,
                    split_st,
                    self.tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    is_train_split=is_train_split,
                    epoch=epoch,
                    seed=self.args.seed,
                    src_dict=self.src_dict,
                    src_bpe_tokenizer=src_bpe_tokenizer,
                    data_type="st"
                )
                data_dict.append(("st",st_data))
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict(
                        data_dict
                )
            )
        else:
            st_infer_data= TripleDatasetCreator.from_tsv(
                self.args.data,
                self.data_cfg,
                split,
                self.tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
                is_train_split=is_train_split,
                epoch=epoch,
                seed=self.args.seed,
                src_dict=self.src_dict,
                src_bpe_tokenizer=src_bpe_tokenizer,
                data_type="st"
            )
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict(
                    [
                        ("st", st_infer_data)
                    ]
                ),
                eval_key=None
                if is_train_split or is_valid_split
                else "st" ,
            )

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextTask, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if TripleDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def _per_task_train_loss(
        self, per_task, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        loss, sample_size, logging_output = criterion(
            model, sample[per_task], per_task
        )
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        from collections import defaultdict

        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)

        # TODO the data in smaple and all_task is not match, the sample may lack of some types data
        for idx, per_task in enumerate(sample.keys()):

            def maybe_no_sync():
                if (
                    self.args.distributed_world_size > 1
                    and hasattr(model, "no_sync")
                    and idx < len(self.all_tasks) - 1
                ):
                    return model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            with maybe_no_sync():
                loss, sample_size, logging_output = self._per_task_train_loss(
                    per_task,
                    model,
                    update_num,
                    criterion,
                    sample,
                    optimizer,
                    ignore_grad,
                )
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                #agg_logging_output[f"{task_pair}:{k}"] += logging_output[k]
                #print(k)
        return agg_loss, agg_sample_size, agg_logging_output

    def _per_task_pair_valid_loss(self, per_task, model, criterion, sample):
        return criterion(model, sample[per_task], per_task)

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            from collections import defaultdict

            agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
            for idx, per_task in enumerate(self.all_tasks):
                if (
                    per_task not in sample
                    or sample[per_task] is None
                    or len(sample[per_task]) == 0
                ):
                    continue
                loss, sample_size, logging_output = self._per_task_pair_valid_loss(per_task, model, criterion, sample)
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                    agg_logging_output[f"{per_task}:{k}"] += logging_output[k]
            #agg_loss, agg_sample_size, agg_logging_output = criterion(model, sample, "st")
        return agg_loss, agg_sample_size, agg_logging_output


    def build_src_bpe(self, args):
        logger.info(f"src tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        #TODO only for speech translation task
        st_infer_data=TripleDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
        return RoundRobinZipDatasets(
            OrderedDict(
                [
                    ("st", st_infer_data)
                ]
            )
        )
        #return SpeechToTextDataset(
        #    "interactive", False, self.data_cfg, src_tokens, src_lengths
        #)
