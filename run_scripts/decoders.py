"""Decode with a channel model.

Usage:
  decoders.py [--dynet-seed SEED] [--dynet-mem MEM] [--dynet-autobatch ON]
  [--transducer=TRANSDUCER] [--sigm2017format] [--no-feat-format]
  [--input=INPUT] [--feat-input=FEAT] [--action-input=ACTION] [--pos-emb] [--avm-feat-format]
  [--compact-feat=COMPACT-FEAT] [--compact-nonlin=COMP-NONLIN]
  [--enc-hidden=HIDDEN] [--dec-hidden=HIDDEN] [--enc-layers=LAYERS] [--dec-layers=LAYERS]
  [--vanilla-lstm] [--mlp=MLP] [--nonlin=NONLIN] [--lucky-w=W]
  [--tag-wraps=WRAPS] [--param-tying] [--verbose=VERBOSE]
  [--decoding-mode=MODE] [--dec-temperature=TEMP] [--dec-sample-size=SSIZE] [--dec-keep-sampling]
  [--result-fn-suffix=SUFFIX]
  TRAIN-PATH DEV-PATH RESULTS-PATH [--test-path=TEST-PATH] [--reload-path=RELOAD-PATH]

Arguments:
  TRAIN-PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV-PATH      path to dataset to decode, possibly relative to "data/all/"
  RESULTS-PATH  results file to be written, possibly relative to "results"

Options:
  -h --help                     show this help message and exit
  --dynet-seed SEED             DyNET seed
  --dynet-mem MEM               allocates MEM bytes for DyNET
  --dynet-autobatch ON          perform automatic minibatching
  --transducer=TRANSDUCER       transducer model to use: haem / hard [default: haem]
  --sigm2017format              assume sigmorphon 2017 input format (lemma, word, feats)
  --no-feat-format              no features format (input, *, output)
  --input=INPUT                 character embedding dimension [default: 100]
  --feat-input=FEAT             feature embedding dimension.  "0" denotes "bag-of-features". [default: 20]
  --action-input=ACTION         action embedding dimension [default: 100]
  --pos-emb                     embedding POS (or the first feature in the sequence of features) as a non-atomic feature
  --avm-feat-format             features are treated as an attribute-value matrix (`=` pairs attributes with values)
  --compact-feat=COMPACT-FEAT   non-linearly map resulting feature vector to this dimension [default: 400]
  --compact-nonlin=COMP-NONLIN  if compact-feat, apply this non-linearity to compact-feat dimensional feature vector.
                                    ReLU/tanh/linear [default: linear]
  --enc-hidden=HIDDEN           hidden layer dimension of encoder RNNs [default: 200]
  --enc-layers=LAYERS           number of layers in encoder RNNs [default: 1]
  --dec-hidden=HIDDEN           hidden layer dimension of decoder RNNs [default: 200]
  --dec-layers=LAYERS           number of layers in decoder RNNs [default: 1]
  --vanilla-lstm                use vanilla LSTM instead of DyNet 1's default coupled LSTM
  --mlp=MLP                     MLP hidden layer dimension. "0" denotes "no hidden layer". [default: 0]
  --nonlin=NONLIN               if mlp, this non-linearity is applied after the hidden layer. ReLU/tanh [default: ReLU]
  --lucky-w=W                   if feat-input==0, scale the "bag-of-features" vector by W [default: 55]
  --param-tying                 use same embeddings for characters and actions inserting them
  --tag-wraps=WRAPS             wrap lemma and word with word boundary tags?
                                  both (use opening and closing tags)/close (only closing tag)/None [default: both]
  --verbose=VERBOSE             verbose==1: print to stdout processing info, verbose==2: visualize results of internal
                                  evaluation, display train and dev set alignments, costs [default: 0]
  --decoding-mode=MODE          Which decoding method to use: sampling/channel [default: channel]
  --dec-temperature=TEMP        If decoding-mode=="sampling", what inverse temperature to use. By default, sample
                                  straight from the model. [default: 1]
  --dec-sample-size=SSIZE       If decoding-mode=="sampling", how many samples to draw per input? [default: 20]
  --dec-keep-sampling           If decoding-mode=="sampling", whether to keep sampling until the required number of
                                  unique samples is produced for each input.
  --result-fn-suffix=SUFFIX     Add the given suffix to the result file. Defaults to the empty string. [default: ]
  --test-path=TEST-PATH         path to test dataset to decode
  --reload-path=RELOAD-PATH     reload a pretrained model at this path (possibly relative to RESULTS-PATH)
"""
from docopt import docopt

import dynet as dy
import numpy as np
import random
import time
import os
#import sys
#import codecs
from trans.args_processor import process_arguments
from trans.datasets import BaseDataSet, action2string
from trans.defaults import UNK
from collections import defaultdict
import json

ENCODING = 'utf8'


def compute_channel(name, batches, transducer, vocab):
    """
    Compute log probabilities (aka channel scores) for pairs of input & output strings plus, possibly, features.
    :param name: Name of the set of batches (dev or test).
    :param batches: Batches.
    :param transducer: Transducer.
    :param vocab: Vocabulary object.
    :return: JSON-serializable dictionary of results.
    """
    then = time.time()
    print('evaluating on {} data...'.format(name))
    output = dict()
    for j, (act_word, batch) in enumerate(batches.items()):
        dy.renew_cg()
        #dy.renew_cg(immediate_compute=True, check_validity=True)
        log_prob = []
        pred_acts = []
        candidates = []
        features = []
        for sample in batch:
            # @TODO one could imagine to draw multiple samples (then sampling=True)....
            feats = sample.pos, sample.feats
            unseen = False
            for a in sample.actions:
                if a >= vocab.act_train:
                    print('Action unseen in training: ', vocab.act.i2w[a], '...skipping this sample.')
                    unseen = True
            if unseen:
                continue
            loss, _, predicted_actions = transducer.transduce(sample.lemma, feats,
                                                              oracle_actions={'loss': "nll",
                                                                              'rollout_mixin_beta': 1.,
                                                                              'global_rollout': False,
                                                                              'target_word': sample.actions,
                                                                              'optimal': True,
                                                                              'bias_inserts': False},
                                                              sampling=False,
                                                              channel=True,
                                                              external_cg=True)
            pred_acts.append(action2string(predicted_actions, vocab))
            log_prob.append(dy.esum(loss).value())  # sum log probabilities of actions
            candidates.append(sample.lemma_str)
            features.append(sample.feat_str)
        results = {'candidates': candidates, 'log_prob': log_prob, 'acts': pred_acts, 'feats': features}
        output[sample.word_str] = results
        if j > 0 and j % 100 == 0:
            print('\t\t...{} batches'.format(j))
    print('\t...finished in {:.3f} sec'.format(time.time() - then))
    return output


def sample(name, batches, inverse_temperature, sample_size, transducer, vocab, keep_sampling_until_sample_size):
    """
    Sample output strings from the model given some input string and possibly features. Possibly, use
    `inverse_temperature` (if approaches 0, then it "flattens" the sampling distribution) and produce exactly
    `sample_size` output strings if `keep_sampling_until_sample_size` is set to True.
    :param name: Name of the set of batches (dev or test).
    :param batches: Batches.
    :param inverse_temperature: Smoothing parameter for the sampling distribution (if 0, then uniform probability;
        if 1., then equals true distribution; if goes to inf, winner-take-all.)
    :param sample_size: Number of output strings or draws per input.
    :param keep_sampling_until_sample_size: Whether to keep sampling until `sample_size` output strings are sampled
        per each input or stop after `sample_size` draws from the model.
    :param transducer: Transducer.
    :param vocab: Vocabulary object.
    :return: JSON-serializable dictionary of results.
    """
    then = time.time()
    print('sampling for {} data...'.format(name))
    print('For each sample, will sample {} {} distinct output sequences. '
          'Using alpha("inverse temperature")={}.'.format(
            "exactly" if keep_sampling_until_sample_size else "at most", sample_size, inverse_temperature))
    output = dict()
    for j, batch in enumerate(batches):
        dy.renew_cg()
        for sample in batch:
            #print(j, 'Sample: ', sample.word_str)
            feats = sample.pos, sample.feats
            log_prob = []
            pred_acts = []
            candidates = []
            features = []
            if keep_sampling_until_sample_size is False:
                counter = sample_size
            else:
                counter = 0
            T = inverse_temperature
            while ((not keep_sampling_until_sample_size or len(candidates) < sample_size) and
                   (keep_sampling_until_sample_size or counter > 0)):
                if counter < 0 and counter % 100 == 0:
                    T -= T / 10
                    #print('Sampling the same things all the time. Decreasing alpha slightly to', T)
                loss, prediction, predicted_actions = \
                    transducer.transduce(sample.lemma, feats, sampling=True,
                                         inverse_temperature=T,
                                         external_cg=True)
                counter -= 1
                if predicted_actions not in pred_acts:
                    log_prob.append(dy.esum(loss).value())  # N.B. not affected by inverse_temperature
                    pred_acts.append(predicted_actions)
                    candidates.append(prediction)
                    features.append(sample.feat_str)
                    #print('Draw: ', prediction, action2string(predicted_actions, vocab))
            results = {'candidates': candidates, 'log_prob': log_prob,
                       'acts': [action2string(pa, vocab) for pa in pred_acts], 'feats': features}
            output[sample.lemma_str] = results
        # report progress
        if j > 0 and j % 50 == 0:
            print('\t\t...{} batches'.format(j))
    print('\t...finished in {:.3f} sec'.format(time.time() - then))
    return output


if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)
    ddoc = docopt(__doc__)
    print('Processing arguments...')
    ddoc.update({'--align-dumb': True, '--mode': 'il', '--sample-weights': False, '--dev-subsample': 0,
                 '--dev-stratify-by-pos' : False, '--try-reverse': False, '--iterations': 0, '--beam-width': 0,
                 '--beam-widths': None, '--dropout': 0, '--pretrain-dropout': False, '--optimization': None, '--l2': 0,
                 '--alpha': 0, '--beta': 0, '--no-baseline': False, '--epochs': 0, '--patience': 0,
                 '--pick-loss': False, '--pretrain-epochs': 0, '--pretrain-until': 0, '--batch-size': 0,
                 '--decbatch-size': 0, '--sample-size': 0, '--scale-negative': 0, '--il-decay': 0, '--il-k': 0,
                 '--il-tforcing-epochs': 0, '--il-loss': 'nll', '--il-bias-inserts': False, '--il-beta': 1,
                 '--il-global-rollout': False, '--il-optimal-oracle': True, 'DECODERS': True})
    print(ddoc)

    arguments = process_arguments(ddoc)
    paths, data_arguments, model_arguments, optim_arguments = arguments

    print('Loading data... Dataset: {}'.format(data_arguments['dataset']))
    # @TODO get rid of train_data entirely: VOCAB must be loaded from file
    train_data = data_arguments['dataset'].from_file(paths['train_path'], reload_path=paths['reload_path'],
                                                     **data_arguments)
    VOCAB = train_data.vocab
    VOCAB.train_cutoff()  # knows that entities before come from train set
    dev_data = data_arguments['dataset'].from_file(paths['dev_path'], vocab=VOCAB, **data_arguments)
    if paths['test_path']:
        print('***TEST DECODING not supported. Run this script with DEV-PATH set to test set.')
    #     # no alignments, hence BaseDataSet
    #     test_data = BaseDataSet.from_file(paths['test_path'], vocab=VOCAB, **data_arguments)
    # else:
    #     test_data = None

    model = dy.Model()
    transducer = model_arguments['transducer'](model, VOCAB, **model_arguments)
    print('Trying to load model from: {}'.format(paths['reload_model_path']))
    model.populate(paths['reload_model_path'])

    if ddoc['--decoding-mode'] == 'channel':
        print('Decoding with a channel model...')

        dev_batches = defaultdict(set)
        for s in dev_data.samples:
            if not any(t.lemma == s.lemma and t.actions == s.actions and t.pos == s.pos and t.feats == s.feats
                       for t in dev_batches[s.word_str]):
                dev_batches[s.word_str].add(s)
        dev_batches = dict(dev_batches)
        print('Total number of dev batches: ', len(dev_batches))

        dev_output = compute_channel('dev', dev_batches, transducer, VOCAB)
        fname = 'dev_channel{}.json'.format(ddoc['--result-fn-suffix'])

    elif ddoc['--decoding-mode'] == 'sampling':
        print('Decoding by sampling...')

        inverse_temperature = float(ddoc['--dec-temperature'])
        sample_size = int(ddoc['--dec-sample-size'])
        keep_sampling_until_sample_size = ddoc['--dec-keep-sampling']
        # normal dev samples
        decbatch_size = 1  # @TODO not making this a command-line parameter: Keep comp. graph as small as possible.
        dev_batches = [dev_data.samples[i:i + decbatch_size] for i in range(0, len(dev_data), decbatch_size)]
        print('Total number of dev batches: ', len(dev_batches))

        dev_output = sample('dev', dev_batches, inverse_temperature, sample_size, transducer, VOCAB,
                            keep_sampling_until_sample_size)
        fname = 'dev_samples{}.json'.format(ddoc['--result-fn-suffix'])

    else:
        raise NotImplementedError('Other decoding methods not supported '
                                  '(for beam-search, use eval mode of run_transducer.py).')

    path = os.path.join(paths['results_file_path'], fname)
    print('Writing results to file "{path}".'.format(path=path))
    with open(path, 'w', encoding=ENCODING) as w:
        json.dump(dev_output, w, indent=4)

    # if test_data:
    #     print('=========TEST EVALUATION:=========')
    #     test_batches = defaultdict(set)
    #     for s in test_data.samples:
    #         test_batches[s.word_str].add(s)
    #     test_batches = dict(test_batches)
    #     compute_channel('test', test_batches, transducer, VOCAB)
