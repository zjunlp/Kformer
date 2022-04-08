# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import torch
from fairseq.data import (
    Dictionary,
    IdDataset,
    ListDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
    encoders,
)
from fairseq.tasks import LegacyFairseqTask, register_task


@register_task("med_qa")
class MedQATask(LegacyFairseqTask):
    """Task to finetune RoBERTa for Social IQA."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data", metavar="DIR", help="path to data directory; we load <split>.jsonl"
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument("--num-classes", type=int, default=4)

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.mask = vocab.add_symbol("<mask>")

        self.bpe = encoders.build_bpe(args)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert (
            args.criterion == "sentence_ranking"
        ), "Must set --criterion=sentence_ranking"

        # load data and label dictionaries
        vocab = cls.load_dictionary(os.path.join(args.data, "dict.txt"))
        print("| dictionary: {} types".format(len(vocab)))

        return cls(args, vocab)

    def load_dataset(
        self, split, epoch=1, combine=False, data_path=None, return_only=False, **kwargs
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def binarize(s, append_bos=False, append_eos=True):
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = self.vocab.encode_line(
                s,
                append_eos=append_eos,
                add_if_not_exist=False,
            ).long()
            if append_bos and self.args.init_token is not None:
                tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
            return tokens

        if data_path is None:
            data_path = os.path.join(self.args.data, split + ".jsonl")
        if not os.path.exists(data_path):
            raise FileNotFoundError("Cannot find data: {}".format(data_path))
        
        src_tokens = [[] for i in range(self.args.num_classes)]
        src_lengths = [[] for i in range(self.args.num_classes)]
        knowledge = [[] for i in range(self.args.num_classes)]
        labels = []
        max_know_length = 256
        with open(data_path) as h:
            for line in h:
                example = json.loads(line.strip())
                if "answer_idx" in example:
                    labels.append(example["answer_idx"])
                question = example["question"]
                assert len(example["choices"]) == self.args.num_classes
                # format: `<s> Q: Where would I not want a fox? </s> A: hen house </s>`
                question = "Q: " + question
                question_toks = binarize(question, append_bos=True)
                for i, choice in enumerate(example["choices"]):
                    src = "A: " + choice["text"]
                    choice_bin = binarize(src)
                    max_len_for_question = 256 - choice_bin.shape[0] - 1
                    # question_bin = question_toks[:max_len_for_question] 
                    knowledges = choice["knowledge"]
                    know_bin = []
                    knowledge_text = ""
                    for know in knowledges:
                        knowledge_text += know 
                    for know in knowledges:
                        bina_know = binarize(know, append_eos=False)
                        # bina_know = binarize(know)
                        # if bina_know.shape[0] > max_know_length:
                        #     print(bina_know.shape[0])
                        bina_know = bina_know[:max_know_length]
                        padding_length = max_know_length - len(bina_know)
                        if padding_length>0:
                            bina_know = torch.cat((bina_know, torch.tensor([self.source_dictionary.pad()] * padding_length, dtype=torch.int64)))
                        know_bin.append(bina_know)
                    knowledge_bin = torch.stack(know_bin)
                    src_bin = torch.cat([question_toks, choice_bin])
                    # src_bin = torch.cat([src_bin,binarize(knowledge_text)])
                    # if (len(src_bin))>512:
                    #     src_bin = src_bin[:511]
                    #     src_bin = torch.cat([src_bin,torch.tensor([2], dtype=torch.int64)])
                    src_tokens[i].append(src_bin)
                    knowledge[i].append(knowledge_bin)
                    src_lengths[i].append(len(src_bin))
        assert all(
            len(src_tokens[0]) == len(src_tokens[i])
            for i in range(self.args.num_classes)
        )
        assert len(src_tokens[0]) == len(src_lengths[0])
        assert len(labels) == 0 or len(labels) == len(src_tokens[0])

        for i in range(self.args.num_classes):
            src_lengths[i] = np.array(src_lengths[i])
            src_tokens[i] = ListDataset(src_tokens[i], src_lengths[i])
            src_lengths[i] = ListDataset(src_lengths[i])
            knowledge[i] = BaseWrapperDataset(knowledge[i])
        dataset = {
            "id": IdDataset(),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens[0], reduce=True),
        }
        for i in range(self.args.num_classes):
            dataset.update(
                {
                    "net_input{}".format(i + 1): {
                        "src_tokens": RightPadDataset(
                            src_tokens[i],
                            pad_idx=self.source_dictionary.pad(),
                        ),
                        "src_lengths": src_lengths[i],
                        "knowledge": knowledge[i],
                    }
                }
            )
        if len(labels) > 0:
            dataset.update({"target": RawLabelDataset(labels)})

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[np.maximum.reduce([src_token.sizes for src_token in src_tokens])],
        )

        with data_utils.numpy_seed(self.args.seed):
            dataset = SortDataset(
                dataset,
                # shuffle
                sort_order=[np.random.permutation(len(dataset))],
            )

        print("| Loaded {} with {} samples".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        model.register_classification_head(
            "sentence_classification_head",
            num_classes=1,
        )

        return model

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab
