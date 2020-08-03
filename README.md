# Code for Paper Imitation Learning for Neural Morphological String Transduction

Peter Makarov and Simon Clematide. 2018. EMNLP

https://arxiv.org/abs/1808.10701

This is a python3 port of the paper's code. It also includes additional features. To install the project in editable mode, do

```
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

This should also compile Mans Hulden's aligner library (`trans/align.c`). If the shared library `trans/libalign.so` has not been created, run the Makefile (`make -C trans`).

All test data reported in the paper can be found [here](https://github.com/ZurichNLP/emnlp2018-imitation-learning-for-neural-morphology-test-data).
