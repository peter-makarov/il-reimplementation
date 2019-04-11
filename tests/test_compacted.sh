#sigm17rudev="tests/data/russian-dev"
#sigm17rurtl="tests/data/russian-train-low"
sigm17rudev="tests/data/sigmorphon-2018-all-at-once/lang-feature/low/low_dev.txt"
sigm17rurtl="tests/data/sigmorphon-2018-all-at-once/lang-feature/low/low_train.txt"
#results="tests/data/SIGM17_RU_LOW"
results="tests/data/LANG_ONE4ALL"
python run_scripts/run_transducer.py --dynet-seed 1 --dynet-mem 500 --dynet-autobatch 0  --transducer=haem --sigm2017format \
--input=100 --feat-input=20 --action-input=100 --pos-emb  --enc-hidden=200 --dec-hidden=200 --enc-layers=1 --compact-feat=200 --compact-nonlin=linear \
--dec-layers=1   --mlp=0 --nonlin=ReLU --il-optimal-oracle --il-loss=nll --il-beta=0.5 --il-global-rollout --verbose=0 --dev-subsample=0.2  --dev-stratify-by-pos \
--dropout=0 --optimization=ADADELTA --l2=0  --batch-size=1 --decbatch-size=25  --patience=15 --epochs=50 \
--tag-wraps=both --param-tying  --mode=il   --beam-width=0 --beam-widths=4  $sigm17rurtl  $sigm17rudev  $results
