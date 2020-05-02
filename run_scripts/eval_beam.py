#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""



{
    "absent": {
        "": {
            "candidates": [
                "a p s ɑ̃",
                "a p s",
                "a e s ɑ̃",
                "a f s ɑ̃",
                "a b ɑ̃",
                "a a s ɑ̃",
                "a i s ɑ̃",
                "a b s ɑ̃",
                "a p ɑ̃",
                "a p s "
            ],
            "log_prob": [
                -0.21450185775756836,
                -1.7277116775512695,
                -6.623240947723389,
                -6.670190811157227,
                -6.79126763343811,
                -6.806300163269043,
                -6.9377121925354,
                -6.99894392490387,
                -7.253797292709351,
                -7.49785715341568
            ],
            "acts": [
                "<COPY><COPY> +Sub(p)+ <COPY> +Sub(ɑ)++Sub(̃)+<DEL>⟫",
                "<COPY><COPY> +Sub(p)+ <COPY><DEL><DEL><DEL>⟫",
                "<COPY><COPY> +Sub(e)+ <COPY> +Sub(ɑ)++Sub(̃)+<DEL>⟫",
                "<COPY><COPY> +Sub(f)+ <COPY> +Sub(ɑ)++Sub(̃)+<DEL>⟫",
                "<COPY><COPY> <COPY> <DEL>+Sub(ɑ)++Sub(̃)+<DEL>⟫",
                "<COPY><COPY> +Sub(a)+ <COPY> +Sub(ɑ)++Sub(̃)+<DEL>⟫",
                "<COPY><COPY> +Sub(i)+ <COPY> +Sub(ɑ)++Sub(̃)+<DEL>⟫",
                "<COPY><COPY> <COPY> <COPY> +Sub(ɑ)++Sub(̃)+<DEL>⟫",
                "<COPY><COPY> +Sub(p)++Sub( )++Sub(ɑ)++Sub(̃)+<DEL>⟫",
                "<COPY><COPY> +Sub(p)+ <COPY> <DEL><DEL><DEL>⟫"
            ],
            "target": "a p s ɑ̃"
        }
    },
    "abîme": {
        "": {
            "candidates": [
                "a b i m",
                "a b î m",
                "a b ɛ m",

"""

__appname__ = "[application name here]"
__author__  = "AA"
__version__ = "0.0pre0"
__license__ = "GNU GPL 3.0 or later"

import logging
log = logging.getLogger(__name__)
import sys
import json

class MainApplication(object):

    def __init__(self, args):
        self.args = args
        self.data = json.load(self.args.infile)

    def run(self):
        bad = []
        offranksum = 0
        offrankcount = 0
        for k in self.data:
            for feature in self.data[k]:
                entry = self.data[k][feature]

                if entry["target"] == entry["candidates"][0]:
                    entry["status"] = "correct"
                else:
                    entry["status"] = "wrong"
                    try:
                        rank = entry["candidates"].index(entry["target"])
                        entry["in_beam"] = rank
                        if rank > 1:
                            offranksum +=rank
                            offrankcount += 1
                    except ValueError:
                        entry["in_beam"] = None
                        bad.append({'tg':entry["target"],'sr':k})
                #print('ENTRY',entry)
        for k in self.data:
            for feature in list(self.data[k]):
                entry = self.data[k][feature]
                if entry["status"] == "correct":
                    del self.data[k][feature]
        for k in list(self.data):
            if self.data[k] == {}:
                del self.data[k]
        self.data['__bad__'] = bad
        self.data['__badcount__'] = len(bad)
        self.data['__offrankcount__'] = offrankcount
        self.data['__offranksum__'] = offranksum
        print(json.dumps(self.data,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    import argparse
    description = ""
    epilog = ""
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-l', '--logfile', dest='logfile',
                      help='write log to FILE', metavar='FILE')
    parser.add_argument('-v', '--verbose', dest='verbose',default=2,type=int, metavar="LEVEL",
                      help='set verbosity level: 0=CRITICAL, 1=ERROR, 2=WARNING, 3=INFO 4=DEBUG (default %(default)s)')
    parser.add_argument(
        "infile",
        metavar="INPUT",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="Input file (default: STDIN)",
    )

    args = parser.parse_args()

    log_levels = [logging.CRITICAL, logging.ERROR, logging.WARNING,
                  logging.INFO, logging.DEBUG]
    logging.basicConfig(level=log_levels[args.verbose],
                        format='%(asctime)-15s %(levelname)s: %(message)s')

    # launching application ...
    MainApplication(args).run()
