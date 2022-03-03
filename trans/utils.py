"""Utility functions and classes."""
from typing import Any, Dict, List, Optional, TextIO, Union
import dataclasses
import logging
import os
import time
import re
import unicodedata
import torch
from trans.vocabulary import PAD


@dataclasses.dataclass
class Sample:
    input: str
    target: Optional[str]
    encoded_input: Optional[List[int]] = None
    optimal_actions: Optional[List[int]] = None


@dataclasses.dataclass
class Samples:
    samples: List[Sample]

    @property
    def input(self):
        return [s.input for s in self.samples]

    @property
    def target(self):
        return [s.target for s in self.samples]

    @property
    def encoded_input(self):
        return [s.encoded_input for s in self.samples]

    @property
    def optimal_actions(self):
        return [s.optimal_actions for s in self.samples]


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

    def get_data_loader(self, **kwargs):
        if 'collate_fn' not in kwargs:
            if 'pad_index' not in kwargs:
                pad_index = PAD

            def collate(batch: List):
                if len(batch) > 1:
                    max_len = len(max(batch, key=lambda b: len(b.input)).encoded_input)
                    for s in batch:
                        s.encoded_input += [pad_index] * (max_len - len(s.encoded_input))
                return Samples(batch)

            kwargs['collate_fn'] = collate

        return torch.utils.data.DataLoader(self, **kwargs)


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
