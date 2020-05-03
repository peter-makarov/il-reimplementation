#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
import os
import sys
import codecs
import collections
import numpy as np
import unicodedata
#from sklearn.metrics import cohen_kappa_score

"""
Ensemble system for result files

"""

# dict maps row number of entry to counter with tuple (LEMMA, WF, MORPH)
SYSTEMS = collections.defaultdict(collections.Counter)

ROWS = collections.Counter()
VOTES = collections.Counter()
TIE_STATS = collections.Counter()
GOLDEN = None
GOLDENDEVSET = None
PREDICTION_COLUMN = 1

def select_nbest(prediction_files, nbest, devset=False):
    """
    Return list with nbest prediction files according to devset statistics
    """
    global GOLDENDEVSET
    accuracies = []
    best = []
    if GOLDENDEVSET is None:
        GOLDENDEVSET = GOLDEN
    if "dev.predictions" in prediction_files[0]:
        devset = True
    for j, file in enumerate(prediction_files):
        corr = 0
        total = 0
        wrong = 0
        if not devset:
            #Identify the file with prediction on the dev set
            if not "test.predictions" in file:
                print("ERROR: Prediction file on test should be of the format *.test.predictions, check the file {}".format(file), file=sys.stderr)
                exit(1)
            file = file.replace("test.predictions", "dev.predictions")
            if not os.path.exists(file):
                print("ERROR: File {} does not exist! Check the input prediction files.".format(file), file=sys.stderr)
                exit(1)
            else:
                print("INFO: READING DEVSET FILE {}".format(file), file=sys.stderr)
        with codecs.open(file, 'r', encoding='utf-8') as f:
            for i,l in enumerate(f):
                total += 1
                l = l.rstrip()
                if GOLDENDEVSET[i] == l.split('\t')[PREDICTION_COLUMN]:
                    corr += 1
                else:
                    wrong += 1
            accuracy = corr/total
            accuracies.append(accuracy)
            print('DEV-ACCURACY:\t%d\t%s\t%f' %(j,file,accuracy), file=sys.stderr)

    #return np.argmax(accuracies)
    ranks = list(np.argsort(accuracies))
    ranks.reverse()
    for i in range(0,min(nbest,len(prediction_files))):
        fileindex = ranks[i]
        best.append(prediction_files[fileindex])
        print('ENSEMBLE-FILE:\t%d\t%s\t%f' %(i,prediction_files[fileindex],accuracies[fileindex]), file=sys.stderr)

    return best


def read_golden(filepath):
    """
    Does not work right now!
    """
    GOLDEN = []
    with codecs.open(filepath,'r',encoding="utf-8") as f:
        for l in f:
            l = unicodedata.normalize('NFKC', l)
            l = l.rstrip().split('\t')
            GOLDEN.append(l[1])
    return GOLDEN

def read_files(args):
    global SYSTEMS, ROWS
    for arg in args:
        with codecs.open(arg,'r',encoding="utf-8") as f:
            for i,l in enumerate(f):
                l = unicodedata.normalize('NFKC', l)
                ROWS[arg] += 1
                SYSTEMS[i][l.strip()] += 1   # l.strip() produces errors!!!!
        print('#INFO, file %s contains %d items'%(arg,ROWS[arg]), file=sys.stderr)
    if len(set(ROWS.values())) > 1:
        print('#WARNING, number of predictions not equal', list(ROWS.values()), file=sys.stderr)

def pprint_ties(distribution, i, options):
    if options.debug:
        print("\t".join('#VOTES:'+str(c) + k for (k,c) in distribution.most_common()), file=sys.stderr)
    else:
        distr = distribution.most_common()
        #print >> sys.stderr, '# GOLDEN', GOLDEN[i], distr[0]
        try:
            top_pred = distr[0][0].split('\t')[1]
        except Exception as e:
            print('# ERROR CAUSING ENTRY', GOLDEN[i], distr[0][0], file=sys.stderr)
            assert len(distr[0][0].split('\t')) == 2  # system might opt to not predict anything if input is corrupted
            top_pred = ''  # this gets lost somewhere in the code...
        if GOLDEN[i] == top_pred:
            label = 'CORRECT'
        else:
            label = 'WRONG'
        if (len(distr) > 1 and distr[0][1] == distr[1][1]) or label == 'WRONG':
            print(label,"[",GOLDEN[i],"]", "\t".join('#VOTES:'+str(c)+" " + k for (k,c) in distribution.most_common()), file=sys.stderr)



def process(options=None,args=None):
    """
    Do the processing
    """
    global GOLDEN, GOLDENDEVSET
    correct = 0
    wrong = 0
    if options.debug:
        print(options, file=sys.stderr)
    if options.golden_file:

        GOLDEN = read_golden(options.golden_file)
    if options.golden_devset:

        GOLDENDEVSET = read_golden(options.golden_devset)
    if options.nbest > 0:
        args = select_nbest(args, options.nbest, devset=False)
        print('INFO: ENSEMBLING FILES', args, file=sys.stderr)
    read_files(args)
    for i in sorted(SYSTEMS):
        nbest = SYSTEMS[i].most_common()  # returns list of item * count pairs [('the', 1143), ('and', 966), ('to', 762), ('of', 669), ('i', 631), ('you', 554),  ('a', 546), ('my', 514), ('hamlet', 471), ('in', 451)]
        if nbest[0][1] == 1:
            TIE_STATS['tie1'] += 1
           # pprint_ties(SYSTEMS[i])
        elif len(nbest) > 1:
            if nbest[0][1] == nbest[1][1]:
                TIE_STATS['tie'+str(nbest[0][1])] += 1
               # pprint_ties(SYSTEMS[i])


            else:
                TIE_STATS['notie'] += 1
        else:
            TIE_STATS['notie'] += 1
        pprint_ties(SYSTEMS[i],i, options)
        ensemble_prediction = nbest[0][0].split('\t')[1]
        if ensemble_prediction == GOLDEN[i]:
            correct += 1
        else:
            wrong += 1
        print(f"nbest{ensemble_prediction} G{GOLDEN[i]}",file=sys.stderr)
        print(nbest[0][0], file=sys.stdout)
    print(f"WER: {wrong/(correct+wrong)}",file=sys.stderr)


    print(TIE_STATS, file=sys.stderr)


def main():
    """
    Invoke this module as a script
    """
    global PREDICTION_COLUMN
    parser = OptionParser(
        usage = '%prog [OPTIONS] SYS1 SYS2 ... SYSN > ENSEMBLE',
        version='%prog 0.99', #
        description='Read files from command line and produce ensemble on stdout',
        epilog='Contact simon.clematide@uzh.ch'
        )
    parser.add_option('-l', '--logfile', dest='logfilename',
                      help='write log to FILE', metavar='FILE')
    parser.add_option('-q', '--quiet',
                      action='store_true', dest='quiet', default=False,
                      help='do not print status messages to stderr')
    parser.add_option('-d', '--debug',
                      action='store_true', dest='debug', default=False,
                      help='print debug information')
    parser.add_option('-n', '--nbest',
                      action='store', dest='nbest', default=0,type=int,
                      help='limit ensembling to n best systems (%default)')
    parser.add_option('-g', '--golden_file',
                      action='store', dest='golden_file', default=None,
                      help='read and process the golden solutions (1 column only)')
    parser.add_option('-G', '--golden_devset',
                      action='store', dest='golden_devset', default=None,
                      help='read and process the golden devset solutions (1 column only)')
    parser.add_option('-m', '--mode',
                      action='store', dest='mode', default='',
                      help='mode of ensembling expressed by character set; G=gold oracle (take gold solution if its in the system) (%default) ')
    parser.add_option('-C', '--column',
                      action='store', dest='column', default=1,type=int,
                      help='index (zero-based) of column containing the prediction (%default) ')

    (options, args) = parser.parse_args()
    PREDICTION_COLUMN = options.column

    if options.debug:
        print("options=",options, file=sys.stderr)


    process(options=options,args=args)


if __name__ == '__main__':
    main()
