
# SC 
# for l in ady dut jpn rum vie arm  bul  fre  geo  gre  hin  hun  ice  kor  lit ; do LNG=${l} bash tests/sub_lang.bash &  done  ; for l in ady dut jpn rum vie  arm  bul  fre  geo  gre  hin  hun  ice  kor  lit ; do NFK="." LNG=${l} bash tests/sub_lang.bash &  done ; wait 

# SC aditional test languages ady dut jap rum vie
# for l in  ; do LNG=${l} bash tests/sub_lang.bash &  done  & for l in ady dut jap rum vie ; do NFK="" LNG=${l} bash tests/sub_lang.bash &  done ; wait 

# small test # for l in arm  bul  fre  geo  gre  hin  hun  ice  kor  lit ; do EPOCHS=2 LNG=${l} ENC_HIDDEN=20 DEC_HIDDEN=20 bash tests/sub_lang.bash &  done ; wait ; 
#   SED_ALIGNER_EM_ITERATIONS=20 SED_ALIGNER_DISCOUNT="-1" LNG=fre bash tests/sub_lang.bash 
#   SED_ALIGNER_EM_ITERATIONS=20 SED_ALIGNER_DISCOUNT="-1" LNG=fre DROPOUT=0.01 NFK="" bash tests/sub_lang.bash 
# SED_ALIGNER_EM_ITERATIONS=20 SED_ALIGNER_DISCOUNT="-10" LNG=fre DROPOUT=0 NFK="" bash tests/sub_lang.bash 
#   SED_ALIGNER_EM_ITERATIONS=20 SED_ALIGNER_DISCOUNT="-1" LNG=fre DROPOUT=0 NFK="" TRAIN__BATCH_SIZE=20 bash tests/sub_lang.bash 

# results-task1-sub/m+100+1+100+200+200+2+2-a20+-0.5-t+0.02+ADADELTA+1+10+60+12-x0/fre./s1/
# RESULTSDIR=results-task1-sub_ret NFK=. TRAIN__DROPOUT=0.2 FEAT_INPUT=1 TRAIN__BATCH_SIZE=20 BATCHSIZE=1 DEC_LAYERS=2 DROPOUT=0.02 ENC_LAYERS=2 SED_ALIGNER_EM_ITERATIONS=20 LNG=fre bash -x tests/sub_lang.bash 

# RESULTSDIR=results-task1 NFK=. TRAIN__DROPOUT=0.05 FEAT_INPUT=1 TRAIN__BATCH_SIZE=5 BATCHSIZE=5 DEC_LAYERS=2 DROPOUT=0.05 ENC_LAYERS=2 SED_ALIGNER_EM_ITERATIONS=20 LNG=fre bash -x tests/sub_lang.bash 

# results-task1-sub/m+100+1+100+200+200+1+1-a20+-0.5-t+0+ADADELTA+1+10+60+12+0-x0/fre./s2/
# RESULTSDIR=results-task1-sub-nodisc  NFK=. DYNET_SEED=2 FEAT_INPUT=1 TRAIN__BATCH_SIZE=40 BATCHSIZE=1 DEC_LAYERS=1 DROPOUT=0 ENC_LAYERS=1 SED_ALIGNER_EM_ITERATIONS=20 SED_ALIGNER_DISCOUNT=-0.5 LNG=fre bash  tests/sub_lang.bash 
# RESULTSDIR=results-task1-sub-nodisc  NFK=. DYNET_SEED=2 FEAT_INPUT=1  BATCHSIZE=1 DEC_LAYERS=1 DROPOUT=0 ENC_LAYERS=1 SED_ALIGNER_EM_ITERATIONS=20 SED_ALIGNER_DISCOUNT=-0.5 LNG=fre bash  tests/sub_lang.bash 
# RESULTSDIR=results-task1-sub-nodisc  NFK=. DYNET_SEED=2 FEAT_INPUT=1  BATCHSIZE=1 DEC_LAYERS=1 DROPOUT=0 ENC_LAYERS=1 SED_ALIGNER_EM_ITERATIONS=1 SED_ALIGNER_DISCOUNT=-0.5 EPOCHS=1 LNG=fre bash  tests/sub_lang.bash 

# if called with ALT_BATCH_SIZE variable set then the training is done with TRAIN__BATCH_SIZE, but the directory does not change 
: "${RESULTSDIR:=results-task1-sub}"
: "${NFK=.nfd.}"   # ${VAR-VALUE} also if VAR is empty
: "${BATCH_SIZE:=1}"
: "${DROPOUT:=0}"
: "${ACTION_INPUT:=100}"
: "${FEAT_INPUT:=1}"
: "${INPUT:=100}"
: "${ENC_HIDDEN:=200}"
: "${DEC_HIDDEN:=200}"
: "${ENC_LAYERS:=1}"
: "${DEC_LAYERS:=1}"
: "${OPTIMIZATION:=ADADELTA}"
: "${IL_K:=12}"
: "${EPOCHS:=60}"
: "${PATIENCE:=10}"
: "${DYNET_SEED:=1}"
: "${VARIANT:=0}"
: "${SED_ALIGNER_EM_ITERATIONS:=20}"
: "${SED_ALIGNER_DISCOUNT:=-0.5}"
: "${PICK_LOSS:=0}"

if test ${PICK_LOSS} = "1" ; then
export PICK_LOSS=1
export PICK_LOSS_OPTION=--pick-loss
fi

LOGFILE=$(mktemp /tmp/il-reimplementation.XXXXXXXXX)

sigm17rudev="tests/sgm2020data/dev/${LNG}_dev${NFK}tsv"
sigm17rurtl="tests/sgm2020data/train/${LNG}_train${NFK}tsv"
#results="results/sub-0.0005-batch1-.2-200-2-1/${LNG}/"
sigm17rutest="tests/sgm2020data/test/${LNG}_test${NFK}tsv"
results="${RESULTSDIR}/m+${INPUT}+${FEAT_INPUT}+${ACTION_INPUT}+${ENC_HIDDEN}+${DEC_HIDDEN}+${ENC_LAYERS}+${DEC_LAYERS}-a${SED_ALIGNER_EM_ITERATIONS}+${SED_ALIGNER_DISCOUNT}-t+${DROPOUT}+${OPTIMIZATION}+${BATCH_SIZE}+${PATIENCE}+${EPOCHS}+${IL_K}+${PICK_LOSS}-x${VARIANT}/${LNG}${NFK}/s${DYNET_SEED}"

: "${TRAIN__BATCH_SIZE=${BATCH_SIZE}}"
: "${TRAIN__DROPOUT=${DROPOUT}}"
: "${TRAIN__PATIENCE=${PATIENCE}}"


echo "tail -f ${LOGFILE} # for watching log for	 ${results} "

if test "${RELOAD}" = "1" ; then
if test -e ${results}/f.model ; then
	RELOADPATH="--reload-path=${results}"
else 
	echo "ERROR: RELOAD MODE, BUT MISSING ${results}/f.model"
	exit 2
fi
fi
python -u run_scripts/run_transducer.py --dynet-seed ${DYNET_SEED} --dynet-mem 1500 --dynet-autobatch 0  --no-feat-format  --transducer=haem --sigm2017format \
--input=${INPUT} --feat-input=${FEAT_INPUT} --compact-feat=1 --compact-nonlin=linear --action-input=${ACTION_INPUT} --pos-emb  --enc-hidden=${ENC_HIDDEN} --dec-hidden=${DEC_HIDDEN} --enc-layers=${ENC_LAYERS} \
--dec-layers=${DEC_LAYERS}   --mlp=0 --nonlin=ReLU --il-optimal-oracle --il-loss=nll --il-beta=0.5 --il-global-rollout \
--dropout=${TRAIN__DROPOUT} --optimization=${OPTIMIZATION} --l2=0  --batch-size=${TRAIN__BATCH_SIZE}  --decbatch-size=25  --patience=${TRAIN__PATIENCE} --epochs=${EPOCHS} \
--tag-wraps=both --param-tying  --mode=il  --il-k=${IL_K} ${PICK_LOSS_OPTION} --beam-width=8 --beam-widths=4,8 --test-path=${sigm17rutest}  ${RELOADPATH} $sigm17rurtl  $sigm17rudev   $results &> ${LOGFILE}

mv ${LOGFILE} ${results}/transducer.log


python run_scripts/decoders.py  --dynet-seed ${DYNET_SEED} --dynet-mem 3000 --dynet-autobatch 1  --transducer=haem  --no-feat-format  --sigm2017format \
--input=${INPUT} --feat-input=${FEAT_INPUT} --compact-feat=1 --compact-nonlin=linear --action-input=${ACTION_INPUT} --pos-emb --enc-hidden=${ENC_HIDDEN} --dec-hidden=${DEC_HIDDEN} --enc-layers=${ENC_LAYERS} --dec-layers=${DEC_LAYERS} \
--tag-wraps=both --param-tying --mlp=0 --nonlin=ReLU --verbose=0 \
--decoding-mode=beam --dec-temperature=1 --dec-sample-size=20 --dec-keep-sampling --beam-width=8 --test-path=${sigm17rutest}\
$sigm17rurtl  $sigm17rudev  $results

python run_scripts/eval_beam.py ${results}/dev_beam.json >  ${results}/dev_beam.wrong.json

# official greedy eval

paste   <(cut -f 2 ${sigm17rudev}) <(cut -f 2 ${results}/f.greedy.dev.predictions) > ${results}/f.greedy.dev.gold.pred.tsv

python ../2020/task1/evaluation/evaluate.py ${results}/f.greedy.dev.gold.pred.tsv > ${results}/f.greedy.dev.gold.pred.eval.txt 2> ${results}/f.greedy.dev.gold.pred.eval.txt.log

# official beam eval

paste   <(cut -f 2 ${sigm17rudev}) <(cut -f 2 ${results}/f.beam4.dev.predictions) > ${results}/f.beam4.dev.gold.pred.tsv

python ../2020/task1/evaluation/evaluate.py ${results}/f.beam4.dev.gold.pred.tsv > ${results}/f.beam4.dev.gold.pred.eval.txt 2> ${results}/f.beam4.dev.gold.pred.eval.txt.log

# official beam eval

paste   <(cut -f 2 ${sigm17rudev}) <(cut -f 2 ${results}/f.beam8.dev.predictions) > ${results}/f.beam8.dev.gold.pred.tsv

python ../2020/task1/evaluation/evaluate.py ${results}/f.beam8.dev.gold.pred.tsv > ${results}/f.beam8.dev.gold.pred.eval.txt 2> ${results}/f.beam8.dev.gold.pred.eval.txt.log

