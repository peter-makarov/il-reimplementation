from typing import List, Tuple

import dynet as dy
import numpy as np

from trans.defaults import COPY, DELETE, END_WORD, UNK, MAX_ACTION_SEQ_LEN
from trans.stack_lstms import Encoder
from trans.datasets import action2string, lemma2string
from trans.transducer import (log_sum_softmax_loss, log_sum_softmax_margin_loss,
                              sample)
from trans import transducer
from trans.optimal_expert import Expert
from trans.actions import Copy, Del, Ins, Sub

NONLINS = {'tanh': dy.tanh, 'ReLU': dy.rectify, 'linear': lambda e: e}


class Transducer(transducer.Transducer):
    def __init__(self, model, vocab, expert: Expert,
                 char_dim=100, action_dim=100, feat_dim=20,
                 enc_hidden_dim=200, enc_layers=1, dec_hidden_dim=200,
                 dec_layers=1,
                 vanilla_lstm=False, mlp_dim=0, nonlin='ReLU', lucky_w=55,
                 double_feats=False, param_tying=False, pos_emb=True,
                 avm_feat_format=False, compact_feat_dim=400,
                 compact_nonlin='linear', **kwargs):

        self.CHAR_DIM = char_dim
        self.ACTION_DIM = action_dim
        self.FEAT_DIM = feat_dim
        self.ENC_HIDDEN_DIM = enc_hidden_dim
        self.ENC_LAYERS = enc_layers
        self.DEC_HIDDEN_DIM = dec_hidden_dim
        self.DEC_LAYERS = dec_layers
        self.LSTM = dy.VanillaLSTMBuilder if vanilla_lstm else dy.CoupledLSTMBuilder
        self.MLP_DIM = mlp_dim
        self.NONLIN = NONLINS[nonlin]
        self.LUCKY_W = lucky_w
        self.double_feats = double_feats
        self.param_tying = param_tying
        self.pos_emb = pos_emb
        self.avm_feat_format = avm_feat_format
        self.COMPACT_FEAT_DIM = compact_feat_dim
        self.COMPACT_NONLIN = NONLINS[compact_nonlin]

        self.vocab = vocab

        # indices separating train elements from dev/test elements
        self.NUM_FEATS = self.vocab.feat_train
        self.NUM_POS = self.vocab.pos_train
        # an enumeration of all encoded insertions
        self.INSERTS = list(
            range(self.vocab.number_specials, self.vocab.act_train)) + [UNK]
        substitution_offset = self.vocab.act_train - self.vocab.number_specials
        self.SUBSTITUTION_MAP = {
            (ins + substitution_offset): ins
            for ins in range(self.vocab.number_specials, self.vocab.act_train)
        }
        self.REVERSE_SUBSTITUTION_MAP = {
            ins: sub for sub, ins in self.SUBSTITUTION_MAP.items()
        }
        self.SUBSTITUTIONS = sorted(self.SUBSTITUTION_MAP)

        self.NUM_ACTS = self.vocab.act_train + len(self.SUBSTITUTIONS)
        self.NUM_CHARS = self.NUM_ACTS

        self.optimal_expert = expert

        # report stats
        print(u'{} actions: {}'.format(self.NUM_ACTS,
                                       u', '.join(self.vocab.act.keys())))
        print(u'{} features: {}'.format(self.NUM_FEATS,
                                        u', '.join(self.vocab.feat.keys())))
        if self.pos_emb:
            print(u'{} POS: {}'.format(self.NUM_POS,
                                       u', '.join(self.vocab.pos.keys())))
        else:
            print('No POS features.')
        print(u'{} lemma chars: {}'.format(self.NUM_CHARS,
                                           u', '.join(self.vocab.char.keys())))

        if self.avm_feat_format:
            self.NUM_FEAT_TYPES = self.vocab.feat_type_train
            print(u'{} feature types: {}'.format(self.NUM_FEAT_TYPES,
                                                 u', '.join(
                                                     self.vocab.feat_type.keys())))
            if self.pos_emb:
                print(
                    'Assuming AVM features, therefore no specialized pos embedding')
                self.pos_emb = False

        self._build_model(model)
        # for printing
        self.hyperparams = {'CHAR_DIM': self.CHAR_DIM,
                            'FEAT_DIM': self.FEAT_DIM,
                            'ACTION_DIM': self.ACTION_DIM if not self.param_tying else self.CHAR_DIM,
                            'ENC_HIDDEN_DIM': self.ENC_HIDDEN_DIM,
                            'ENC_LAYERS': self.ENC_LAYERS,
                            'DEC_HIDDEN_DIM': self.DEC_HIDDEN_DIM,
                            'DEC_LAYERS': self.DEC_LAYERS,
                            'LSTM': self.LSTM,
                            'MLP_DIM': self.MLP_DIM,
                            'NONLIN': self.NONLIN,
                            'PARAM_TYING': self.param_tying,
                            'POS_EMB': self.pos_emb,
                            'AVM_FEATS': self.avm_feat_format,
                            'COMPACT_FEATS_DIM': self.COMPACT_FEAT_DIM,
                            'COMPACT_NINLIN': self.COMPACT_NONLIN}

    def _optimal_expert_score(self, x: str, t: str, i: int, y: str):

        action_scores = self.optimal_expert.score(x, t, i, y)
        remapped_action_scores = dict()
        for action, score in action_scores.items():
            if isinstance(action, Del):
                remapped_action = DELETE
            elif isinstance(action, Ins):
                remapped_action = self.vocab.act.w2i[action.new]
            elif isinstance(action, Copy):
                remapped_action = COPY
            elif isinstance(action, Sub):
                corresponding_insert = self.vocab.act.w2i[action.new]
                remapped_action = self.REVERSE_SUBSTITUTION_MAP[
                    corresponding_insert]
            elif action == END_WORD:
                remapped_action = action
            else:
                raise ValueError(
                    f"""
                    Unknown action: {action, score}
                    action_scores={action_scores}
                    x={x}
                    t={t}
                    i={i}
                    y={y}
                    """
                )
            remapped_action_scores[remapped_action] = -score
        # print(
        #     f"""
        #     x={x}
        #     t={t}
        #     i={i}
        #     y={y}
        #     action_scores: {action_scores}
        #     remapped_action_scores: {remapped_action_scores}
        #     """
        # )
        return remapped_action_scores

    def expert_rollout(
            self, word: List[str], target_word: List[int],
            rest_of_input: List[int], valid_actions: List[int]
    ) -> Tuple[List[int], np.array]:

        """
        Rolls out with optimal expert policy.

        :param word: The current prediction (y), characters.
        :param target_word: Target string (t), action integer codes.
        :param rest_of_input: Input suffix (x[i:]), lemma integer codes.
        :param valid_actions: Valid actions, action integer codes.
        :return: List of optimal actions (as integer codes) and regrets (
            by convention, invalid actions are set to -inf).
        """

        # normalize all representations
        assert rest_of_input and rest_of_input[-1] == END_WORD, (
            # ignore the end-of-word symbol
            f"""
            rest_of_input: {lemma2string(rest_of_input, self.vocab)}
            """
        )
        x = lemma2string(rest_of_input[:-1], self.vocab)
        i = 0  # because x is a suffix
        t = action2string(target_word, self.vocab).replace("<UNK>", "☭")
        y = "".join(word)

        action_scores = self._optimal_expert_score(x, t, i, y)
        optimal_action_cost = min(action_scores.values())
        worst_valid_regret = max(action_scores.values()) - optimal_action_cost

        accuracy_error_cost = worst_valid_regret + 5  # by convention
        valid_action_set = set(valid_actions)
        regrets = np.ones(self.NUM_ACTS) * accuracy_error_cost
        for action_index, score in action_scores.items():
            try:
                regrets[action_index] = score - optimal_action_cost
            except IndexError as e:
                print(
                    f"""
                    error: {e}
                    x: {x},
                    t: {t},
                    y: {y},
                    action_index: {action_index}
                    action_scores: {action_scores}
                    """
                )
            assert action_index in valid_action_set or action_index == END_WORD, (
                f"""
                action_index: {action_index},
                valid_actions: {valid_action_set},
                action_scores: {action_scores} 
                """
            )
        optimal_actions = list(np.where(regrets == 0)[0])
        assert optimal_actions, (
            f"""
            x: {x},
            t: {t},
            y: {y},
            action_index: {regrets}
            action_scores: {action_scores}
            """
        )

        # by convention, set externally established invalid actions to -inf
        valid_action_mask = np.zeros(regrets.shape, dtype=np.bool_)
        valid_action_mask[valid_actions] = True
        regrets[~valid_action_mask] = -np.inf

        # print(
        #     f"""
        #     rest_of_input: {rest_of_input}
        #     x: {x}
        #     word=y: {word}
        #     target_word: {target_word}
        #     t: {action2string(target_word, self.vocab)}
        #     optimal_actions: {optimal_actions}
        #     costs: {regrets}
        #     """
        #     # actions as strings: {[action2string([a], self.vocab) for a in optimal_actions]}
        # )
        return optimal_actions, regrets

    def _valid_actions(self, encoder):
        valid_actions = [END_WORD]  # allows to always terminate
        valid_actions += self.INSERTS
        if len(encoder) > 1:
            valid_actions.extend([COPY, DELETE])
            valid_actions += self.SUBSTITUTIONS
        return valid_actions

    def transduce(self, lemma, feats, oracle_actions=None, external_cg=True,
                  sampling=False,
                  unk_avg=True, verbose=False, channel=False,
                  inverse_temperature=1.):
        """
        Transduce an encoded lemma and features.
        Args:
            lemma: The input lemma, a list of integer character codes.
            feats: The features determining the morphological transformation. The most common
                   format is a list of integer codes, one code per feature-value pair.
            oracle_actions: `None` means prediction.
                            List of action codes is a static oracle.
                            A dictionary of keys (explained below) is the config for a dynamic oracle.
                                * "target_word": List of action codes for the target word form.
                                * "loss": Which loss function to use (softmax-margin, NLL, MSE).
                                * "rollout_mixin_beta": How to mix reference and learned roll-outs
                                    (1 is only reference, 0 is only model).
                                * "global_rollout": Whether to use one type of roll-out (expert or model)
                                    at the sequence level.
                                * "optimal": Whether to use an optimal or noisy (=buggy) expert
                                * "bias_inserts": Whether to use a buggy roll-out for inserts
                                    (which makes them as cheap as copies)
            external_cg: Whether or not an external computation graph is defined.
            sampling: Whether or not sampling should be used for decoding (e.g. for MRT) or
                      training (e.g. dynamic oracles with exploration / learned roll-ins).
            dynamic: Whether or not `oracle_actions` is a static oracle (list of actions) or a confuguration
                     for a dynamic oracle.
            unk_avg: Whether or not to average all char embeddings to produce UNK embedding
                     (see `self._build_lemma`).
            channel: Used as channel model.
            inverse_temperature: Smoothing parameter for the sampling distribution (if 0, then uniform probability;
                if 1., then equals true distribution; if goes to inf, winner-take-all.)
                Smith & Eisner 2006. "Minimum risk annealing for training log-linear models." In COLING/ACL.
            verbose: Whether or not to report on processing steps.
        """

        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)
        if not external_cg:
            dy.renew_cg()

        dynamic = None  # indicates prediction or static

        if oracle_actions:
            # if not, then prediction
            if isinstance(oracle_actions, dict):
                # dynamic oracle:
                # @TODO NB target word is not wrapped in boundary tags
                target_word = oracle_actions['target_word']
                generation_errors = set()
                dynamic = oracle_actions
            else:
                # static oracle:
                # reverse to enable simple popping
                oracle_actions = oracle_actions[::-1]
                oracle_actions.pop()  # COPY of BEGIN_WORD_CHAR

        # vectorize lemma
        lemma_enc = self._build_lemma(lemma, unk_avg, is_training=(
                    bool(oracle_actions) and not channel))

        # vectorize features
        features = self._build_features(*feats)

        # add encoder and decoder to computation graph
        encoder = Encoder(self.fbuffRNN, self.bbuffRNN)
        decoder = self.wordRNN.initial_state()

        # add classifier to computation graph
        if self.MLP_DIM:
            # decoder output to hidden
            W_s2h = dy.parameter(self.pW_s2h)
            b_s2h = dy.parameter(self.pb_s2h)
        # hidden to action
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)

        # encoder is a stack which pops lemma characters and their
        # representations from the top. Thus, to get lemma characters
        # in the right order, the lemma has to be reversed.
        encoder.transduce(lemma_enc, lemma)

        encoder.pop()  # BEGIN_WORD_CHAR
        action_history = [COPY]
        word = []
        losses = []

        if verbose and not dynamic:
            count = 0
            print()
            print(action2string(oracle_actions, self.vocab))
            print(lemma2string(lemma, self.vocab))

        while len(action_history) <= MAX_ACTION_SEQ_LEN:

            if verbose and not dynamic:
                print('Action: ', count, self.vocab.act.i2w[action_history[-1]])
                print('Encoder length, char: ', lemma, len(encoder),
                      self.vocab.char.i2w[encoder.s[-1][-1]])
                print('word: ', u''.join(word))
                print(('Remaining actions: ', oracle_actions,
                       action2string(oracle_actions, self.vocab)))
                count += 1

            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            valid_actions = self._valid_actions(encoder)
            encoder_embedding = encoder.embedding()
            # decoder
            decoder_input = dy.concatenate([encoder_embedding,
                                            features,
                                            self.ACT_LOOKUP[action_history[-1]]
                                            ])
            decoder = decoder.add_input(decoder_input)
            # classifier
            if self.double_feats:
                classifier_input = dy.concatenate([decoder.output(), features])
            else:
                classifier_input = decoder.output()
            if self.MLP_DIM:
                h = self.NONLIN(W_s2h * classifier_input + b_s2h)
            else:
                h = classifier_input
            logits = W_act * h + b_act
            # get action (argmax, sampling, or use oracle actions)
            if oracle_actions is None:
                # predicting by argmax or sampling
                log_probs = dy.log_softmax(logits, valid_actions)
                log_probs_np = log_probs.npvalue()
                if sampling:
                    action = sample(log_probs_np, inverse_temperature)
                else:
                    action = np.argmax(log_probs_np)
                losses.append(dy.pick(log_probs, action))
            elif dynamic:
                # training with dynamic oracle
                optim_actions, costs = self.expert_rollout(
                    word, target_word, encoder.get_extra(), valid_actions)

                # print(
                #     f"""
                #     optimal_actions2: {optim_actions2}
                #     costs2: {costs2}
                #     """)

                log_probs = dy.log_softmax(logits, valid_actions)
                log_probs_np = log_probs.npvalue()
                if sampling == 1. or np.random.rand() <= sampling:
                    # action is picked by sampling
                    action = sample(log_probs_np)
                    # @TODO IL learned roll-ins are done with policy i.e. greedy / beam search decoding
                    if verbose: print('Rolling in with model: ', action,
                                      self.vocab.act.i2w[action])
                else:
                    # action is picked from optim_actions
                    action = optim_actions[
                        np.argmax([log_probs_np[a] for a in optim_actions])]
                    # print [log_probs_np[a] for a in optim_actions]
                # loss is over all optimal actions.

                if dynamic['loss'] == 'softmax-margin':
                    loss = log_sum_softmax_margin_loss(optim_actions, logits,
                                                       self.NUM_ACTS,
                                                       costs=costs,
                                                       valid_actions=None,
                                                       verbose=verbose)
                elif dynamic['loss'] == 'nll':
                    loss = log_sum_softmax_loss(optim_actions, logits,
                                                self.NUM_ACTS,
                                                valid_actions=valid_actions,
                                                verbose=verbose)
                else:
                    raise NotImplementedError
                if np.isclose(abs(loss.npvalue()), np.inf):
                    print(
                        f"""
                        action: {action}
                        log_probs_np: {log_probs_np}
                        optim_actions: {optim_actions}
                        """
                    )
                losses.append(loss)
                # print 'Action'
                # print action
                # print self.vocab.act.i2w[action]
            else:
                # training with static oracle
                action = oracle_actions.pop()
                log_probs = dy.log_softmax(logits, valid_actions)
                losses.append(dy.pick(log_probs, action))

            action_history.append(action)

            # print 'action, log_probs: ', action, self.vocab.act.i2w[action], losses[-1].scalar_value(), log_probs.npvalue()

            # execute the action to update the transducer state
            if action == COPY:
                # 1. Increment attention index
                try:
                    char_ = encoder.pop()
                except IndexError as e:
                    print(np.exp(log_probs.npvalue()))
                    print('COPY: ', action)
                # 2. Append copied character to the output word
                word.append(self.vocab.char.i2w[char_])
            elif action == DELETE:
                # 1. Increment attention index
                try:
                    encoder.pop()
                except IndexError as e:
                    print(np.exp(log_probs.npvalue()))
                    print('DELETE: ', action)
            elif action == END_WORD:
                # 1. Finish transduction
                break
            elif action in self.SUBSTITUTION_MAP:
                encoder.pop()
                corresponding_insert = self.SUBSTITUTION_MAP[action]
                char_ = self.vocab.act.i2w[corresponding_insert]
                word.append(char_)
            else:
                # one of the INSERT actions
                assert action in self.INSERTS
                # 1. Append inserted character to the output word
                char_ = self.vocab.act.i2w[action]
                word.append(char_)

        word = u''.join(word)

        return losses, word, action_history


    def beam_search_decode(self, lemma, feats, external_cg=True, unk_avg=True,
                           beam_width=4):
        # Returns an expression of the loss for the sequence of actions.
        # (that is, the oracle_actions if present or the predicted sequence otherwise)

        if not external_cg:
            dy.renew_cg()

        # vectorize lemma
        lemma_enc = self._build_lemma(lemma, unk_avg, is_training=False)

        # vectorize features
        features = self._build_features(*feats)

        # add encoder and decoder to computation graph
        encoder = Encoder(self.fbuffRNN, self.bbuffRNN)
        decoder = self.wordRNN.initial_state()

        # encoder is a stack which pops lemma characters and their
        # representations from the top.
        encoder.transduce(lemma_enc, lemma)

        # add classifier to computation graph
        if self.MLP_DIM:
            # decoder output to hidden
            W_s2h = dy.parameter(self.pW_s2h)
            b_s2h = dy.parameter(self.pb_s2h)
        # hidden to action
        W_act = dy.parameter(self.pW_act)
        b_act = dy.parameter(self.pb_act)

        encoder.pop()  # BEGIN_WORD_CHAR

        # a list of tuples:
        #    (decoder state, encoder state, list of previous actions,
        #     log prob of previous actions, log prob of previous actions as dynet object,
        #     word generated so far)
        beam = [(decoder, encoder, [COPY], 0., 0., [])]

        beam_length = 0
        complete_hypotheses = []

        while beam_length <= MAX_ACTION_SEQ_LEN:

            if not beam or beam_width == 0:
                break

            # if show_oracle_actions:
            #    print 'Action: ', count, self.vocab.act.i2w[action_history[-1]]
            #    print 'Encoder length, char: ', lemma, len(encoder), self.vocab.char.i2w[encoder.s[-1][-1]]
            #    print 'word: ', u''.join(word)
            #    print 'Remaining actions: ', oracle_actions, u''.join([self.vocab.act.i2w[a] for a in oracle_actions])
            #    count += 1
            # elif action_history[-1] >= self.NUM_ACTS:
            #    print 'Will be adding unseen act embedding: ', self.vocab.act.i2w[action_history[-1]]

            # compute probability of each of the actions and choose an action
            # either from the oracle or if there is no oracle, based on the model
            expansion = []
            # print 'Beam length: ', beam_length
            for decoder, encoder, prev_actions, log_p, log_p_expr, word in beam:
                # print 'Expansion: ', action2string(prev_actions, self.vocab), log_p, ''.join(word)
                valid_actions = self._valid_actions(encoder)
                # decoder
                decoder_input = dy.concatenate([encoder.embedding(),
                                                features,
                                                self.ACT_LOOKUP[
                                                    prev_actions[-1]]
                                                ])
                decoder = decoder.add_input(decoder_input)
                # classifier
                if self.double_feats:
                    classifier_input = dy.concatenate(
                        [decoder.output(), features])
                else:
                    classifier_input = decoder.output()
                if self.MLP_DIM:
                    h = self.NONLIN(W_s2h * classifier_input + b_s2h)
                else:
                    h = classifier_input
                logits = W_act * h + b_act
                log_probs_expr = dy.log_softmax(logits, valid_actions)
                log_probs = log_probs_expr.npvalue()
                # python3: conversion from np.int64 to int !!!
                top_actions = [int(a) for a in
                               np.argsort(log_probs)[-beam_width:]]
                # print('top_actions: ', top_actions, type(top_actions), type(top_actions[0]),
                #      action2string(top_actions, self.vocab))
                # print('log_probs: ', log_probs)
                # print
                expansion.extend((
                    (decoder, encoder.copy(),
                     list(prev_actions), a, log_p + log_probs[a],
                     log_p_expr + log_probs_expr[a], list(word)) for a in
                top_actions))

            # print 'Overall, {} expansions'.format(len(expansion))
            beam = []
            expansion.sort(key=lambda e: e[4])
            for e in expansion[-beam_width:]:
                decoder, encoder, prev_actions, action, log_p, log_p_expr, word = e

                prev_actions.append(action)

                # execute the action to update the transducer state
                if action == END_WORD:
                    # 1. Finish transduction:
                    #  * beam width should be decremented
                    #  * expansion should be taken off the beam and
                    # stored to final hypotheses set
                    beam_width -= 1
                    complete_hypotheses.append(
                        (log_p, log_p_expr, u''.join(word), prev_actions))
                else:
                    if action == COPY:
                        # 1. Increment attention index
                        char_ = encoder.pop()
                        # 2. Append copied character to the output word
                        word.append(self.vocab.char.i2w[char_])
                    elif action == DELETE:
                        # 1. Increment attention index
                        encoder.pop()
                    elif action in self.SUBSTITUTION_MAP:
                        encoder.pop()
                        corresponding_insert = self.SUBSTITUTION_MAP[action]
                        char_ = self.vocab.act.i2w[corresponding_insert]
                        word.append(char_)
                    else:
                        # one of the INSERT actions
                        assert action in self.INSERTS
                        # 1. Append inserted character to the output word
                        char_ = self.vocab.act.i2w[action]
                        word.append(char_)
                    beam.append((decoder, encoder, prev_actions, log_p,
                                 log_p_expr, word))

            beam_length += 1

        if not complete_hypotheses:
            # nothing found because the model is so crappy
            complete_hypotheses = [
                (log_p, log_p_expr, u''.join(word), prev_actions)
                for _, _, prev_actions, log_p, log_p_expr, word in beam]

        complete_hypotheses.sort(key=lambda h: h[0], reverse=True)
        # print u'Complete hypotheses:'
        # for log_p, _, word, actions in complete_hypotheses:
        #    print u'Actions {}, word {}, log p {:.3f}'.format(action2string(actions, self.vocab), word, log_p)

        return complete_hypotheses