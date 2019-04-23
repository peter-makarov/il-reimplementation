sigm17rudev="tests/data/russian-dev"
sigm17rurtl="tests/data/russian-train-medium"
results="tests/data/SIGM17_RU_MEDIUM"
echo "********** SAMPLING **********"
python run_scripts/decoders.py --dynet-seed 1 --dynet-mem 3000 --dynet-autobatch 1  --transducer=haem --sigm2017format \
--input=100 --feat-input=20 --action-input=100 --pos-emb --enc-hidden=200 --dec-hidden=200 --enc-layers=1 --dec-layers=1 \
--tag-wraps=both --param-tying --compact-feat=0 --compact-nonlin=linear --mlp=0 --nonlin=ReLU --verbose=0 \
--decoding-mode=sampling --dec-temperature=1 --dec-sample-size=20 --dec-keep-sampling \
$sigm17rurtl  $sigm17rudev  $results


echo "********** CHANNEL **********"
python run_scripts/decoders.py --dynet-seed 1 --dynet-mem 3000 --dynet-autobatch 1  --transducer=haem --sigm2017format \
--input=100 --feat-input=20 --action-input=100 --pos-emb --enc-hidden=200 --dec-hidden=200 --enc-layers=1 --dec-layers=1 \
--tag-wraps=both --param-tying --compact-feat=0 --compact-nonlin=linear --mlp=0 --nonlin=ReLU --verbose=0 \
--decoding-mode=channel --dec-temperature=1 --dec-sample-size=20 --dec-keep-sampling \
$sigm17rurtl  $sigm17rudev  $results