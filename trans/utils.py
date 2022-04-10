"""Utility functions and classes."""
from typing import Any, Dict, List, Optional, TextIO, Union
import dataclasses
import logging
import os
import time
import re
import unicodedata
import torch
import pickle
from trans.vocabulary import PAD


@dataclasses.dataclass
class Sample:
    input: str
    target: Optional[str]
    encoded_input: Optional[torch.tensor] = None
    action_history: Optional[torch.tensor] = None
    alignment_history: Optional[torch.tensor] = None
    optimal_actions_mask: Optional[torch.tensor] = None
    valid_actions_mask: Optional[torch.tensor] = None


@dataclasses.dataclass
class TrainingBatch:
    encoded_input: torch.tensor
    action_history: torch.tensor
    alignment_history: torch.tensor
    optimal_actions_mask: torch.tensor
    valid_actions_mask: torch.tensor


@dataclasses.dataclass
class EvalBatch:
    input: List[str]
    target: Optional[List[str]]
    encoded_input: torch.tensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples: Optional[List[Sample]] = None):
        self.samples = samples if samples else []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, id_) -> Sample:
        return self.samples[id_]

    def add_samples(self, samples: Union[List[Sample], Sample]):
        if isinstance(samples, list):
            self.samples.extend(samples)
        else:
            self.samples.append(samples)

    def get_data_loader(self, is_training: bool = False, **kwargs):
        if 'collate_fn' not in kwargs:
            if 'pad_index' not in kwargs:
                pad_index = PAD

            if is_training:
                def collate(batch: List):
                    max_len = len(max([s.encoded_input for s in batch], key=len))-2
                    return TrainingBatch(
                        torch.nn.utils.rnn.pad_sequence([s.encoded_input for s in batch],
                                                        batch_first=True,
                                                        padding_value=pad_index),
                        torch.nn.utils.rnn.pad_sequence([s.action_history for s in batch],
                                                        padding_value=pad_index),
                        torch.nn.utils.rnn.pad_sequence([s.alignment_history for s in batch],
                                                        batch_first=True,
                                                        padding_value=max_len).view(-1),
                        torch.nn.utils.rnn.pad_sequence([s.optimal_actions_mask for s in batch],
                                                        padding_value=False),
                        torch.nn.utils.rnn.pad_sequence([s.valid_actions_mask for s in batch],
                                                        padding_value=False)
                    )
            else:
                def collate(batch: List):
                    return EvalBatch([s.input for s in batch],
                                     [s.target for s in batch],
                                     torch.nn.utils.rnn.pad_sequence([s.encoded_input for s in batch],
                                                                     batch_first=True,
                                                                     padding_value=pad_index)
                                     )

            kwargs['collate_fn'] = collate

        return torch.utils.data.DataLoader(self, **kwargs)

    def persist(self, filename: str):
        with open(filename, mode="wb") as w:
            pickle.dump(self.samples, w)
        logging.info("Wrote precomputed training data to %s.", filename)

    @classmethod
    def from_pickle(cls, path2pkl: str):
        logging.info("Loading precomputed training data from file: %s", path2pkl)
        with open(path2pkl, "rb") as w:
            params: List[Sample] = pickle.load(w)
        return cls(params)


@dataclasses.dataclass
class DecodingOutput:
    accuracy: float
    loss: float
    predictions: List[str]


class OpenNormalize:

    def __init__(self, filename: str, normalize: bool, mode: str = "rt"):
        self.filename = filename
        self.file: Optional[TextIO] = None
        mode_pattern = re.compile(r"[arw]t?$")
        if not mode_pattern.match(mode):
            raise ValueError(f"Unexpected mode {mode_pattern.pattern}: {mode}.")
        self.mode = mode
        if normalize:
            form = "NFD" if self.mode.startswith("r") else "NFC"
            self.normalize = lambda line: unicodedata.normalize(form, line)
        else:
            self.normalize = lambda line: line

    def __enter__(self):
        self.file = open(self.filename, mode=self.mode, encoding="utf8")
        return self

    def __iter__(self):
        for line in self.file:
            yield self.normalize(line)

    def write(self, line: str):
        if not isinstance(line, str):
            raise ValueError(
                f"Line is not a unicode string ({type(line)}): {line}")
        return self.file.write(self.normalize(line))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


def write_results(accuracy: float, predictions: List[str], output: str,
                  normalize: bool, dataset_name: str, beam_width: int = 1,
                  decoding_name: Optional[str] = None,
                  dargs: Dict[str, Any] = None):
    logging.info("%s set accuracy: %.4f.", dataset_name.title(), accuracy)

    if decoding_name is None:
        decoding_name = "greedy" if beam_width == 1 else f"beam{beam_width}"

    eval_file = os.path.join(output, f"{dataset_name}_{decoding_name}.eval")

    with open(eval_file, mode="w") as w:
        if dargs is not None:
            for key, value in dargs.items():
                w.write(f"{key}: {value}\n")
        w.write(f"{dataset_name} accuracy: {accuracy:.4f}\n")

    predictions_tsv = os.path.join(
        output, f"{dataset_name}_{decoding_name}.predictions")

    with OpenNormalize(predictions_tsv, normalize, mode="w") as w:
        w.write("\n".join(predictions))


class Timer:

    def __init__(self):
        self.time = None

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info("\t...finished in %.3f sec.", time.time() - self.time)
