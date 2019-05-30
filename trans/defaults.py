import os

SRC_PATH = os.path.dirname(__file__)
RESULTS_PATH = os.path.join(SRC_PATH, '..')
DATA_PATH = os.path.join(SRC_PATH, '..')
EVALM_PATH = os.path.join(SRC_PATH, '../eval_scripts/evalm.py')

ALIGN_SYMBOL = u'\u3030'  # this is passed to external aligner and has to be a single unicode character (double ~)

### UNK: characters, actions, features
UNK = 0
UNK_CHAR = '<UNK>'

### Word boundary: characters, actions
BEGIN_WORD = 1
END_WORD = 2

BEGIN_WORD_CHAR = u'\u27ea'  # this is passed to external aligner (<)
END_WORD_CHAR = u'\u27eb'  # this is passed to external aligner (>)

### Special actions
STEP = DELETE = 3
COPY = 4
STEP_CHAR = '<STEP>'
DELETE_CHAR = '<DEL>'
COPY_CHAR = '<COPY>'

# all special characters
SPECIAL_CHARS = (ALIGN_SYMBOL, BEGIN_WORD_CHAR, END_WORD_CHAR, UNK_CHAR,
                 STEP_CHAR, DELETE_CHAR, COPY_CHAR)

### trainer defaults
MAX_ACTION_SEQ_LEN = 150
SANITY_SIZE = 100

### for docopt argument processing
NULL_ARGS = 'None', 'none', 'no', '0'

LOCAL_CONFIG_FILE = SRC_PATH + '/defaults_config.py'
if os.path.isfile(LOCAL_CONFIG_FILE):
    try:
        with open(LOCAL_CONFIG_FILE, encoding="utf-8") as f:
            code = compile(f.read(), LOCAL_CONFIG_FILE, 'exec')
            exec(code)
        #            execfile(LOCAL_CONFIG_FILE,globals())
        print('defaults updated from file {LOCAL_CONFIG_FILE} to {globals}'.format(LOCAL_CONFIG_FILE=LOCAL_CONFIG_FILE,
                                                                                   globals=globals()))
    except:
        print('WARNING: could not update defaults from file {LOCAL_CONFIG_FILE}'.format(
            LOCAL_CONFIG_FILE=LOCAL_CONFIG_FILE))
        raise
