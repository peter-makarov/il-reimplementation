"""Decode with a channel model.

Usage:
  decoders.py [--dynet-seed SEED] [--dynet-mem MEM] [--dynet-autobatch ON]
  [--transducer=TRANSDUCER] [--sigm2017format] [--no-feat-format]
  [--input=INPUT] [--feat-input=FEAT] [--action-input=ACTION] [--pos-emb] [--avm-feat-format]
  [--enc-hidden=HIDDEN] [--dec-hidden=HIDDEN] [--enc-layers=LAYERS] [--dec-layers=LAYERS]
  [--vanilla-lstm] [--mlp=MLP] [--nonlin=NONLIN] [--lucky-w=W]
  [--tag-wraps=WRAPS] [--param-tying] [--verbose=VERBOSE]
  TRAIN-PATH DEV-PATH RESULTS-PATH [--test-path=TEST-PATH] [--reload-path=RELOAD-PATH]

Arguments:
  TRAIN-PATH    destination path, possibly relative to "data/all/", e.g. task1/albanian-train-low
  DEV-PATH      development set path, possibly relative to "data/all/"
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
  --test-path=TEST-PATH         test set path
  --reload-path=RELOAD-PATH     reload a pretrained model at this path (possibly relative to RESULTS-PATH)
"""

from __future__ import division
from docopt import docopt

import dynet as dy
import numpy as np
import random
import time
import os
import sys
import codecs
from args_processor import process_arguments
from datasets import BaseDataSet, action2string
from collections import defaultdict
import json

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


def compute_channel(name, batches, transducer, vocab, paths, encoding='utf8'):
    then = time.time()
    print 'evaluating on {} data...'.format(name)
    output = dict()
    for j, (act_word, batch) in enumerate(batches.items()):
        dy.renew_cg()
        log_prob = []
        pred_acts = []
        candidates = []
        for sample in batch:
            # @TODO one could imagine to draw multiple samples (then sampling=True)....
            feats = sample.pos, sample.feats
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
            pred_acts.append(action2string(predicted_actions, vocab).encode(encoding))
            log_prob.append(dy.esum(loss).value())  # sum log probabilities of actions
            candidates.append(sample.lemma_str.encode(encoding))
        results = {'candidates': candidates, 'log_prob': log_prob, 'acts': pred_acts}
        output[(sample.word_str).encode(encoding)] = results
        if j > 0 and j % 100 == 0:
            print '\t\t...{} batches'.format(j)
    print '\t...finished in {:.3f} sec'.format(time.time() - then)

    path = os.path.join(paths['results_file_path'], name + '_channel.json')
    print 'Writing results to file "{path}".'.format(path=path)
    with open(path, 'w') as w:
        json.dump(output, w, indent=4)
    return output


if __name__ == "__main__":

    np.random.seed(42)
    random.seed(42)

    ddoc = docopt(__doc__)
    print ddoc

    print 'Processing arguments...'
    ddoc.update({'--align-dumb': True, '--mode': 'il', '--try-reverse': False, '--iterations': 0, '--beam-width': 0,
                 '--beam-widths': None, '--dropout': 0, '--pretrain-dropout': False, '--optimization': None, '--l2': 0,
                 '--alpha': 0, '--beta': 0, '--no-baseline': False, '--epochs': 0, '--patience': 0,
                 '--pick-loss': False, '--pretrain-epochs': 0, '--pretrain-until': 0, '--batch-size': 0,
                 '--decbatch-size': 0, '--sample-size': 0, '--scale-negative': 0, '--il-decay': 0, '--il-k': 0,
                 '--il-tforcing-epochs': 0, '--il-loss': 'nll', '--il-bias-inserts': False, '--il-beta': 1,
                 '--il-global-rollout': False, '--il-optimal-oracle': True})
    arguments = process_arguments(ddoc)
    paths, data_arguments, model_arguments, optim_arguments = arguments

    print 'Loading data... Dataset: {}'.format(data_arguments['dataset'])
    train_data = data_arguments['dataset'].from_file(paths['train_path'], **data_arguments)
    VOCAB = train_data.vocab
    VOCAB.train_cutoff()  # knows that entities before come from train set
    dev_data = data_arguments['dataset'].from_file(paths['dev_path'], vocab=VOCAB, **data_arguments)
    if paths['test_path']:
        # no alignments, hence BaseDataSet
        test_data = BaseDataSet.from_file(paths['test_path'], vocab=VOCAB, **data_arguments)
    else:
        test_data = None

    dev_batches = defaultdict(set)
    for s in dev_data.samples:
        dev_batches[s.word_str].add(s)
    dev_batches = dict(dev_batches)

    model = dy.Model()
    transducer = model_arguments['transducer'](model, VOCAB, **model_arguments)
    print 'Trying to load model from: {}'.format(paths['reload_path'])
    model.populate(paths['reload_path'])
    compute_channel('dev', dev_batches, transducer, VOCAB, paths)
    if test_data:
        print '=========TEST EVALUATION:========='
        test_batches = defaultdict(set)
        for s in test_data.samples:
            test_batches[s.word_str].add(s)
        test_batches = dict(test_batches)
        compute_channel('test', test_batches, transducer, VOCAB, paths)
