import json
import os
from collections import Counter
from typing import Dict

from trans.defaults import (UNK, UNK_CHAR, BEGIN_WORD, BEGIN_WORD_CHAR,
    END_WORD, END_WORD_CHAR, STEP, STEP_CHAR, DELETE, DELETE_CHAR,
    COPY, COPY_CHAR)

#############################################################
# VOCABULARY
#############################################################

class Vocab(object):
    def __init__(self, w2i=None, encoding='utf8'):
        if w2i is None: 
            self.w2i = dict()
        else:
            self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in self.w2i.items()}
        self.encoding = encoding
        self.freqs = Counter(self.i2w.keys())

    @classmethod
    def from_list(cls, words, encoding='utf8'):
        w2i = {}
        idx = 0
        for word in set(words):
            if encoding:
                word = word.decode(encoding)
            w2i[word] = idx
            idx += 1
        return Vocab(w2i, encoding=encoding)

    @classmethod
    def from_json(cls, d: Dict):
        v = Vocab(w2i=d['w2i'], encoding=d['encoding'])
        v.freqs = Counter()
        v.freqs.update(d['freqs'])
        return v
    
    def __getitem__(self, word):
        # encodes the word if it is not in vocab
        if self.encoding:
            word = word.decode(self.encoding)
        if word in self.w2i:
            idx = self.w2i[word]
        else:
            idx = self.size()
            self.w2i[word] = idx
            self.i2w[idx] = word
        self.freqs[idx] += 1
        return idx
    
    def __contains__(self, word):
        if self.encoding:
            word = word.decode(self.encoding)
        return word in self.w2i
    
    def keys(self): return self.w2i.keys()
    
    def freq(self): return dict(self.freqs)
    
    def __repr__(self): return str(self.w2i)

    def __len__(self): return self.size()
    
    def size(self): return len(self.w2i.keys())

    def serialize(self) -> Dict:
        return dict(w2i=self.w2i, encoding=self.encoding, freqs=self.freqs)

    

class VocabBox(object):
    def __init__(self, acts, pos_emb, avm_feat_format, param_tying, encoding):

        self.w2i_acts = dict(acts)
        self.encoding = encoding
        self.pos_emb = pos_emb
        self.avm_feat_format = avm_feat_format
        self.param_tying = param_tying
        self.act = Vocab(acts, encoding=encoding)
        # number of special actions
        self.number_specials = len(self.w2i_acts)
        # special features
        w2i_feats = {UNK_CHAR : UNK}
        self.feat = Vocab(w2i_feats, encoding=encoding)
        if pos_emb:
            # pos features get special treatment
            self.pos = Vocab({UNK_CHAR : UNK}, encoding=encoding)
            print('VOCAB will index POS separately.')
        else:
            self.pos = self.feat
        if avm_feat_format:
            # feature types get encoded, too
            self.feat_type = Vocab(dict(), encoding=encoding)
            print('VOCAB will index all feature types.')
        else:
            self.feat_type = self.feat
        if param_tying:
            # use one set of indices for acts and chars
            self.char = self.act
            print('VOCAB will use same indices for actions and chars.')
        else:
            # special chars
            w2i_chars = {BEGIN_WORD_CHAR : BEGIN_WORD,
                         END_WORD_CHAR : END_WORD,
                         UNK_CHAR : UNK}
            self.char = Vocab(w2i_chars, encoding=encoding)
        # encoding of words
        self.word = Vocab(encoding=encoding)
        # training set cut-offs
        self.act_train  = None
        self.feat_train = None
        self.pos_train  = None
        self.char_train = None
        self.feat_type_train = None
    def __repr__(self):
        return ('VocabBox (act, feat, pos, char, feat_type) with the following '
                'special actions: {}'.format(self.w2i_acts))

    def train_cutoff(self, path=None):
        # store indices separating training set elements
        # from elements encoded later from unseen samples
        if self.act_train is None:  # the rest are None too
            self.act_train  = len(self.act)
            self.feat_train = len(self.feat)
            self.pos_train  = len(self.pos)
            self.char_train = len(self.char)
            self.feat_type_train = len(self.feat_type)
        else:
            print('Train cutoff has already been applied.')
        if path:
            path2vocab = os.path.join(path, 'vocab.json')
            if os.path.exists(path2vocab):
                print('Path to vocab exists. Will not rewrite vocab: ', path2vocab)
            else:
                print('Writing vocab to: ', path2vocab)
                with open(path2vocab, 'w') as w:
                    json.dump(self.serialize(), w, indent=4, ensure_ascii=False)

    def serialize(self) -> Dict:
        return dict(encoding=self.encoding, pos_emb=self.pos_emb,
                    avm_feat_format=self.avm_feat_format, param_tying=self.param_tying,
                    w2i_acts=self.w2i_acts, number_specials=self.number_specials,
                    act=self.act.serialize(), char=self.char.serialize(), word=self.word.serialize(),
                    feat=self.feat.serialize(), pos=self.pos.serialize(), feat_type=self.feat_type.serialize())

class MinimalVocab(VocabBox):
    def __init__(self, pos_emb=True, avm_feat_format=False, param_tying=False, encoding=None):
        acts = {UNK_CHAR : UNK,
                BEGIN_WORD_CHAR : BEGIN_WORD,
                END_WORD_CHAR : END_WORD,
                STEP_CHAR : STEP}
        super(MinimalVocab, self).__init__(acts, pos_emb, avm_feat_format, param_tying, encoding)

    @classmethod
    def from_json(cls, d: Dict):
        v = cls(d['pos_emb'], d['avm_feat_format'], d['param_tying'], d['encoding'])
        assert v.w2i_acts == d['w2i_acts']
        v.act = Vocab.from_json(d['act'])
        v.feat = Vocab.from_json(d['feat'])
        v.word = Vocab.from_json(d['word'])
        v.pos = Vocab.from_json(d['pos']) if d['pos_emb'] else v.feat
        v.feat_type = Vocab.from_json(d['feat_type']) if d['avm_feat_format'] else v.feat
        v.char = v.act if d['param_tying'] else Vocab.from_json(d['char'])
        v.train_cutoff()
        return v


class EditVocab(VocabBox):
    def __init__(self, pos_emb=True, avm_feat_format=False, param_tying=False, encoding=None):
        acts = {UNK_CHAR : UNK,
                BEGIN_WORD_CHAR : BEGIN_WORD,
                END_WORD_CHAR : END_WORD,
                DELETE_CHAR : DELETE,
                COPY_CHAR : COPY}
        super(EditVocab, self).__init__(acts, pos_emb, avm_feat_format, param_tying, encoding)

    @classmethod
    def from_json(cls, d: Dict):
        v = cls(d['pos_emb'], d['avm_feat_format'], d['param_tying'], d['encoding'])
        assert v.w2i_acts == d['w2i_acts']
        v.act = Vocab.from_json(d['act'])
        v.feat = Vocab.from_json(d['feat'])
        v.word = Vocab.from_json(d['word'])
        v.pos = Vocab.from_json(d['pos']) if d['pos_emb'] else v.feat
        v.feat_type = Vocab.from_json(d['feat_type']) if d['avm_feat_format'] else v.feat
        v.char = v.act if d['param_tying'] else Vocab.from_json(d['char'])
        v.train_cutoff()
        return v