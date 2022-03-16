"""Unit tests for transducer.py."""
import unittest
import argparse

import numpy as np
from scipy.special import log_softmax

import torch

from trans import optimal_expert
from trans import transducer
from trans import vocabulary
from trans.actions import Copy, ConditionalCopy, ConditionalDel, \
    ConditionalIns, ConditionalSub, Sub


np.random.seed(1)


class TransducerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        vocabulary_ = vocabulary.Vocabularies()
        vocabulary_.encode_input("foo")
        vocabulary_.encode_actions("bar")
        expert = optimal_expert.OptimalExpert()

        args = argparse.Namespace(
            device='cpu',
            char_dim=100,
            action_dim=100,
            enc_type='lstm',
            enc_hidden_dim=200,
            enc_layers=1,
            enc_bidirectional=True,
            enc_dropout=0.,
            dec_hidden_dim=100,
            dec_layers=1
        )
        cls.transducer = transducer.Transducer(
            vocabulary_, expert, args)

    def test_sample(self):
        log_probs = log_softmax([5, 4, 10, 1])
        action_code = self.transducer.sample(log_probs)
        self.assertTrue(0 <= action_code < self.transducer.number_actions)

    def test_compute_valid_actions(self):
        valid_actions = self.transducer.compute_valid_actions(3)
        self.assertTrue(self.transducer.number_actions, len(valid_actions))
        valid_actions = self.transducer.compute_valid_actions(1)
        self.assertTrue(not valid_actions[vocabulary.COPY])

    def test_remap_actions(self):
        action_scores = {Copy("w", "w"): 7., Sub("w", "v"): 5.}
        expected = {ConditionalCopy(): 7., ConditionalSub("v"): 5.}
        remapped = self.transducer.remap_actions(action_scores)
        self.assertDictEqual(expected, remapped)

    def test_expert_rollout(self):
        optimal_actions = self.transducer.expert_rollout(
            input_="foo", target="bar", alignment=1, prediction=["b", "a"])
        expected = {self.transducer.vocab.encode_unseen_action(a)
                    for a in (ConditionalIns("r"), ConditionalDel())}
        self.assertSetEqual(expected, set(optimal_actions))


if __name__ == "__main__":
    TransducerTests().run()
