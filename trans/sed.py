"""
Based on Ristad and Yianilos (1998) Learning String Edit Distance.
(https://www.researchgate.net/publication/3192848_Learning_String_Edit_Distance)
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,\
    Union

import abc
import math
import multiprocessing
import numbers
import os
import pickle

import numpy as np
from scipy.special import logsumexp


from trans.actions import Aligner, Sub, Del, Ins


ParamDict = Union[
    Dict[Union[Tuple[str, str], str], float],
    float]

LARGE_NEG_CONST = -float(10 ** 6)
TOL = 10 ** -10

EOS = '<#>'  # end of sequence symbol
UNK = '࿋'  # '<unk>'


class StochasticEditDistance(Aligner):

    def __init__(self, source_alphabet: Iterable[Any],
                 target_alphabet: Iterable[Any],
                 param_dicts: Optional[str] = None,
                 smart_init: Optional[bool] = False,
                 copy_proba: float = 0.9,
                 discount: float = 10 ** -5,
                 *args, **kwargs) -> None:
        """
        Implementation of the Stochastic Edit Distance model from Ristad & Yianilos 1998 "Learning String Edit
        Distance". The model is a memoryless probabilistic weighted finite-state transducer that by use of character
        edits (insertions, deletions, substitutions) maps one string to another. Edit weights are learned with
        Expectation-Maximization.

        :param source_alphabet: Characters of all input strings.
        :param target_alphabet: Characters of all target strings.
        :param param_dicts: Dictionaries of learned parameters.
        :param smart_init: Initialization of parameters with bias towards copying.
        :param copy_proba: If smart_init, how much mass to give to copy edits.
        :param discount: Pseudocount for assigning non-zero probability to unknown edit (to meaningfully handle OOV
            words).
        """
        self.param_dicts = param_dicts
        self.smart_init = smart_init
        self.EOS = EOS
        if param_dicts:
            # load dicts from file
            self.from_pickle(self.param_dicts)
        else:
            # build WFST topology from training data
            self.build_sed(source_alphabet, target_alphabet, discount)

            if self.smart_init:
                self.initialize_smart(copy_proba)
            else:
                self.initialize_random()

        expected_number_of_edits = (
                len(self.delta_sub) + len(self.delta_del) + len(
            self.delta_ins) + 1)
        assert expected_number_of_edits == self.N, (
            expected_number_of_edits, self.N)
        assert np.isclose(0., logsumexp(
            list(self.delta_sub.values()) + list(self.delta_del.values()) +
            list(self.delta_ins.values()) + [self.delta_eos]))

    def build_sed(self,
                  source_alphabet: Iterable[Any],
                  target_alphabet: Iterable[Any],
                  discount: Optional[float] = None):

        self.source_alphabet: Set[Any] = set(source_alphabet)
        self.target_alphabet: Set[Any] = set(target_alphabet)

        self.len_source_alphabet = len(self.source_alphabet)
        self.len_target_alphabet = len(self.target_alphabet)

        # all edits
        self.N = self.len_source_alphabet * self.len_target_alphabet + \
                 self.len_source_alphabet + self.len_target_alphabet + 1

        if discount is None:
            self.default = discount
        else:
            self.default = \
                np.log(discount / self.N)  # log probability of unseen edits

        self.discount = discount

    def initialize_smart(self, copy_proba: float):
        """Initializes weights with a strong bias to copying."""

        if not (0 < copy_proba < 1):
            raise ValueError(
                f'0 < copy probability={copy_proba} < 1 doesn\'t hold.'
            )
        num_copy_edits = len(self.target_alphabet & self.source_alphabet)
        num_rest = self.N - num_copy_edits
        # params for computing word confusion probability
        alpha = np.log(copy_proba / num_copy_edits)  # copy log probability
        log_p = np.log((1 - copy_proba) / num_rest)  # log probability of
        # substitution, deletion, and insertion
        self._initialize(alpha, log_p)

    def initialize_random(self):
        """Initializes weights uniformly."""

        uniform_weight = np.log(1 / self.N)
        self._initialize(uniform_weight, uniform_weight)

    def _initialize(self, log_copy_proba: float, log_rest_proba: float):
        """Initializes weights.

        For simplicity, ignores discount in initialization.
        """

        self.delta_sub = {(s, t): log_copy_proba if s == t else log_rest_proba
                          for s in self.source_alphabet
                          for t in self.target_alphabet}
        self.delta_del = {s: log_rest_proba for s in self.source_alphabet}
        self.delta_ins = {t: log_rest_proba for t in self.target_alphabet}
        self.delta_eos = log_rest_proba

    def from_pickle(self, path2pkl: str) -> None:
        """
        Load delta* parameters from pickle.
        :param path2pkl: Path to pickle.
        """
        try:
            print('Loading sed channel parameters from file: ', path2pkl)
            with open(path2pkl, 'rb') as w:
                (self.delta_sub, self.delta_del, self.delta_ins,
                 self.delta_eos, discount) = pickle.load(w)

            self.build_sed(
                source_alphabet=self.delta_del.keys(),
                target_alphabet=self.delta_ins.keys(),
                discount=discount
            )

            # some random sanity checks
            assert all((s, t) in self.delta_sub for s in self.source_alphabet
                       for t in self.target_alphabet)
            assert (len(self.delta_sub) ==
                    len(self.source_alphabet) * len(self.target_alphabet))
            assert isinstance(self.delta_eos, numbers.Real)
        except (OSError, KeyError, AssertionError) as e:
            print(
                f'"{path2pkl}" exists and data in right format?',
                os.path.exists(path2pkl)
            )
            raise e

    def forward_evaluate(self, source: Sequence,
                         target: Sequence) -> np.ndarray:
        """
        Compute dynamic programming table (in log real) filled with forward log probabilities.
        :param source: Source string.
        :param target: Target string.
        :return: Dynamic programming table.
        """
        T, V = len(source), len(target)
        alpha = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        alpha[0, 0] = 0.
        for t in range(T + 1):
            for v in range(V + 1):
                summands = [alpha[t, v]]
                if v > 0:
                    summands.append(
                        self.delta_ins.get(target[v - 1], self.default) +
                        alpha[t, v - 1]
                    )
                if t > 0:
                    summands.append(
                        self.delta_del.get(source[t - 1], self.default) +
                        alpha[t - 1, v]
                    )
                if v > 0 and t > 0:
                    summands.append(
                        self.delta_sub.get(
                            (source[t - 1], target[v - 1]), self.default
                        ) + alpha[t - 1, v - 1]
                    )
                alpha[t, v] = logsumexp(summands)
        alpha[T, V] += self.delta_eos
        return alpha

    def backward_evaluate(self, source: Sequence,
                          target: Sequence) -> np.ndarray:
        """
        Compute dynamic programming table (in log real) filled with backward log probabilities (the probabilities of
        the suffix, i.e. p(source[t:], target[v:]) e.g. p('', 'a') = p(ins(a))*p(#).
        :param source: Source string.
        :param target: Target string.
        :return: Dynamic programming table.
        """
        T, V = len(source), len(target)
        beta = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        beta[T, V] = self.delta_eos
        for t in range(T, -1, -1):
            for v in range(V, -1, -1):
                summands = [beta[t, v]]
                if v < V:
                    summands.append(
                        self.delta_ins.get(target[v], self.default) +
                        beta[t, v + 1]
                    )
                if t < T:
                    summands.append(
                        self.delta_del.get(source[t], self.default) +
                        beta[t + 1, v]
                    )
                if v < V and t < T:
                    summands.append(
                        self.delta_sub.get(
                            (source[t], target[v]), self.default) +
                        beta[t + 1, v + 1]
                    )
                beta[t, v] = logsumexp(summands)
        return beta

    def ll(self, sources: Sequence, targets: Sequence,
           weights: Sequence[float] = None,
           decode_threads: Optional[int] = None):
        """
        Computes weighted log likelihood.
        :param sources: Source strings.
        :param targets: Target strings.
        :param weights: Weights for pairs of source-target strings.
        :param decode_threads: Speed up the for loop.
        :return: Weighted log likelihood.
        """
        if decode_threads and decode_threads > 1:

            data_len = len(sources)
            step_size = math.ceil(data_len / decode_threads)
            print(
                f'Will compute weighted likelihood in {decode_threads} chunks '
                'of size {step_size}'
            )
            pool = multiprocessing.Pool()
            grouped_samples = [(sources[i:i + step_size],
                                targets[i:i + step_size],
                                weights[i:i + step_size])
                               for i in range(0, data_len, step_size)]
            results = pool.map(self._weighted_forward, grouped_samples)
            pool.terminate()
            pool.join()
            ll = np.mean([w for ww in results for w in ww])
        else:
            # single thread computation
            ll = np.mean([weight * self.forward_evaluate(source, target)[-1, -1]
                          for source, target, weight in
                          zip(sources, targets, weights)])
        return ll

    def _weighted_forward(self, ss_tt_ww: Tuple[List, List, List]):
        """
        Helper function for parallelized computation of weighted log likelihood.
        :param ss_tt_ww: Tuple of sources, targets, weights.
        :return: Weighted log likelihood of the samples.
        """
        ss, tt, ww = ss_tt_ww
        return [w * self.forward_evaluate(s, t)[-1, -1] for s, t, w in
                zip(ss, tt, ww)]

    class Gammas:
        def __init__(self, sed: 'StochasticEditDistance'):
            """
            Container for non-normalized probabilities.
            :param sed: Channel.
            """
            self.sed = sed
            self.eos = 0
            self.sub = {k: 0. for k in self.sed.delta_sub}
            self.del_ = {k: 0. for k in self.sed.delta_del}
            self.ins = {k: 0. for k in self.sed.delta_ins}

        def normalize(self):
            """
            Normalize probabilities and assign them to the channel's deltas.
            :param discount: Unnormalized quantity for edits unseen in training.
            """
            # all mass to distribute among edits
            denom = np.log(
                self.eos + sum(self.del_.values()) + sum(self.ins.values()) +
                sum(self.sub.values()) + self.sed.discount * self.sed.N
            )
            self.sub = {
                k: np.log(self.sub[k] + self.sed.discount) - denom
                for k in self.sub
            }
            self.del_ = {
                k: np.log(self.del_[k] + self.sed.discount) - denom
                for k in self.del_
            }
            self.ins = {
                k: np.log(self.ins[k] + self.sed.discount) - denom
                for k in self.ins
            }
            self.eos = np.log(self.eos + self.sed.discount) - denom

            assert len(self.sub) + len(self.del_) + len(
                self.ins) + 1 == self.sed.N
            check_sum = logsumexp(
                list(self.sub.values()) + list(self.del_.values()) +
                list(self.ins.values()) + [self.eos]
            )
            assert np.isclose(0., check_sum), check_sum
            # set the channel's delta to log normalized gammas
            self.sed.delta_eos = self.eos
            self.sed.delta_sub = self.sub
            self.sed.delta_ins = self.ins
            self.sed.delta_del = self.del_

    def em(self, sources: Sequence, targets: Sequence,
           weights: Optional[Sequence[float]] = None,
           iterations: int = 10, decode_threads: Optional[int] = None,
           test: bool = True,
           verbose: bool = False) -> None:
        """
        Update the channel parameter's delta* with Expectation-Maximization.
        :param sources: Source strings.
        :param targets: Target strings.
        :param weights: Weights for the pairs of source and target strings.
        :param iterations: Number of iterations of EM.
        :param decode_threads: Number of threads to use for the computation of log-likelihood if test is True.
        :param test: Whether to report log-likelihood.
        :param verbose: Verbosity.
        """
        if weights is None:
            weights = [1.] * len(sources)
        if test:
            print('Initial weighted LL=',
                  self.ll(sources, targets, weights, decode_threads=0))
        # print('SED bulkscore cache: ', self.bulkscore.cache_info())  # -> cache and multiprocessing don't work together
        for i in range(iterations):
            gammas = self.Gammas(
                self)  # container for unnormalized probabilities
            for sample_num, (source, target, weight) in enumerate(
                    zip(sources, targets, weights)):
                if weight < TOL:
                    if verbose:
                        print('Weight is below TOL. Skipping: ', source, target,
                              weight)
                    continue
                self.expectation_step(source, target, gammas, weight)
                if test and sample_num > 0 and sample_num % 1000 == 0:
                    print(f'\t...processed {sample_num} samples')
            gammas.normalize()  # maximization step: normalize and assign to self.delta*
            # self.bulkscore.cache_clear()
            # print('SED bulkscore cache cleared: ', self.bulkscore.cache_info())
            if test:
                print('IT_{}='.format(i),
                      self.ll(sources, targets, weights, decode_threads=0))

    def expectation_step(self, source: Sequence, target: Sequence,
                         gammas: Gammas, weight: float = 1.) -> None:
        """
        Accumumate soft counts.
        :param source: Source string.
        :param target: Target string.
        :param gammas: Unnormalized probabilities that we are learning.
        :param weight: Weight for the pair (`source`, `target`).
        """
        alpha = np.exp(self.forward_evaluate(source, target))
        beta = np.exp(self.backward_evaluate(source, target))
        gammas.eos += weight
        T, V = len(source), len(target)
        for t in range(T + 1):
            for v in range(V + 1):
                # (alpha = probability of prefix) * probability of edit * (beta = probability of suffix)
                rest = beta[t, v] / alpha[T, V]
                if t > 0:
                    gammas.del_[source[t - 1]] += \
                        weight * alpha[t - 1, v] * np.exp(
                            self.delta_del[source[t - 1]]) * rest
                if v > 0:
                    gammas.ins[target[v - 1]] += \
                        weight * alpha[t, v - 1] * np.exp(
                            self.delta_ins[target[v - 1]]) * rest
                if t > 0 and v > 0:
                    gammas.sub[(source[t - 1], target[v - 1])] += \
                        weight * alpha[t - 1, v - 1] * np.exp(
                            self.delta_sub[(source[t - 1], target[v - 1])]
                        ) * rest

    def viterbi_distance(self, source: Sequence, target: Sequence,
                         with_alignment: bool = False) -> \
            Union[float, Tuple[List, float]]:
        """
        Viterbi edit distance \propto max_{edits} p(target, edit | source).
        :param source: Source string.
        :param target: Target string.
        :param with_alignment: Whether to output the corresponding sequence of edits.
        :return: Probability score and, optionally, the sequence of edits that gives this score.
        """
        T, V = len(source), len(target)
        alpha = np.full((T + 1, V + 1), LARGE_NEG_CONST)
        alpha[0, 0] = 0.
        for t in range(T + 1):
            for v in range(V + 1):
                alternatives = [alpha[t, v]]
                if v > 0:
                    alternatives.append(
                        self.delta_ins.get(target[v - 1], self.default) +
                        alpha[t, v - 1])
                if t > 0:
                    alternatives.append(
                        self.delta_del.get(source[t - 1], self.default) +
                        alpha[t - 1, v])
                if v > 0 and t > 0:
                    alternatives.append(
                        self.delta_sub.get((source[t - 1], target[v - 1]),
                                           self.default) + alpha[t - 1, v - 1])
                alpha[t, v] = max(alternatives)
        alpha[T, V] += self.delta_eos
        optim_score = alpha[T, V]
        if not with_alignment:
            return optim_score
        else:
            # compute an optimal alignment
            alignment = []
            ind_w, ind_c = len(source), len(target)
            while ind_w >= 0 and ind_c >= 0:
                if ind_w == 0 and ind_c == 0:
                    return alignment[::-1], optim_score
                elif ind_w == 0:
                    # can only go left, i.e. via insertions
                    ind_c -= ind_c
                    alignment.append(
                        Ins(target[ind_c]))  # minus 1 is due to offset
                elif ind_c == 0:
                    # can only go up, i.e. via deletions
                    ind_w -= ind_w
                    alignment.append(
                        Del(source[ind_w]))  # minus 1 is due to offset
                else:
                    # pick the smallest cost actions
                    pind_w = ind_w - 1
                    pind_c = ind_c - 1
                    action_idx = np.argmax([alpha[pind_w, pind_c],
                                            alpha[ind_w, pind_c],
                                            alpha[pind_w, ind_c]])
                    if action_idx == 0:
                        action = Sub(source[pind_w], target[pind_c])
                        ind_w = pind_w
                        ind_c = pind_c
                    elif action_idx == 1:
                        action = Ins(target[pind_c])
                        ind_c = pind_c
                    else:
                        action = Del(source[pind_w])
                        ind_w = pind_w
                    alignment.append(action)

    def stochastic_distance(self, source: Sequence, target: Sequence) -> float:
        """
        Stochastic edit distance \propto sum_{edits} p(target, edit | source) = p(target | source)
        :param source: Source string.
        :param target: Target string.
        :return: Probability score.
        """
        return self.forward_evaluate(source, target)[-1, -1]

    def to_pickle(self, pkl: str) -> None:
        """
        Write parameters delta* to file.
        :param pkl: The pickle filename.
        """
        with open(pkl, 'wb') as w:
            pickle.dump((self.delta_sub, self.delta_del, self.delta_ins,
                         self.delta_eos, self.discount), w)

    def save_model(self, path2model: str) -> None:
        return self.to_pickle(path2model)

    def update_model(self, sources: Sequence, targets: Sequence,
                     weights: Optional[Sequence[float]] = None,
                     em_iterations: int = 10,
                     decode_threads: Optional[int] = None,
                     test: bool = True, verbose: bool = False,
                     output_path: Optional[str] = "/tmp", **kwargs) -> None:
        """
        Update channel model parameters by maximizing weighted likelihood by Expectation-Maximization.
        :param sources: Source strings.
        :param targets: Target strings.
        :param weights: Weights for the pairs of source and target strings.
        :param em_iterations: Number of iterations of EM.
        :param decode_threads: Number of threads to use for the computation of log-likelihood if test is True.
        :param test: Whether to report log-likelihood.
        :param verbose: Verbosity.
        :param output_path: Path where to write learned weights.
        """
        if test:
            print(
                f'Updating channel model parameters by maximizing weighted '
                f'likelihood using EM ({em_iterations} iterations).'
            )
        self.em(sources=sources, targets=targets, weights=weights,
                iterations=em_iterations, decode_threads=decode_threads,
                test=test, verbose=verbose)

        path2pkl = os.path.join(output_path, 'param_dict.pkl')
        self.save_model(path2pkl)
        print(f'Wrote latest model weights to "{path2pkl}".')

    @classmethod
    def fit_from_data(cls, lines: Iterable[str],
                      smart_init: bool = False, em_iterations: int = 30):

        source_alphabet = set()
        target_alphabet = set()
        sources = []
        targets = []
        for line in lines:
            line = line.strip()
            if line:
                source, target, *rest = line.split("\t")
                source_alphabet.update(source)
                target_alphabet.update(target)
                sources.append(source)
                targets.append(target)

        sed = cls(source_alphabet, target_alphabet, smart_init=smart_init)
        sed.update_model(sources, targets, em_iterations=em_iterations)
        return sed

    @property
    def parameter_dictionaries(self) -> Dict[str, ParamDict]:

        return {
            "substitutions": self.delta_sub,
            "insertions": self.delta_ins,
            "deletions": self.delta_del,
            "eos": self.delta_eos
        }

    def action_sequence_cost(self, x: Sequence[Any], y: Sequence[Any],
                             x_offset: int, y_offset: int) -> float:

        return self.viterbi_distance(source=x[x_offset:], target=y[y_offset:])

    def action_cost(self, action: Any) -> float:
        if isinstance(action, Del):
            return self.delta_del.get(action.old, self.default)
        elif isinstance(action, Ins):
            return self.delta_ins.get(action.new, self.default)
        elif isinstance(action, Sub):
            return self.delta_sub.get(
                (action.old, action.new), self.default)
        elif isinstance(action, int):
            return self.delta_eos
        else:
            return self.delta_sub.get(
                (action.old, action.old), self.default)

