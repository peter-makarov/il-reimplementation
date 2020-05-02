SHELL:=/bin/bash
export SHELLOPTS:=errexit:pipefail
.SECONDARY:


# not to be modified varied
RESULTSDIR?=results-task1-sub

# 
INPUT_SET ?= 100    # 200 is worse
FEAT_INPUT_SET ?= 1
ACTION_INPUT_SET ?= 100  # 200 is worse
ENC_HIDDEN_SET ?= 200
DEC_HIDDEN_SET ?= 200
ENC_LAYERS_SET ?= 1
DEC_LAYERS_SET ?= 1
SED_ALIGNER_EM_ITERATIONS_SET ?= 20
SED_ALIGNER_DISCOUNT_SET ?= -0.5
DROPOUT_SET ?= 0
OPTIMIZATION_SET ?= ADADELTA
BATCH_SIZE_SET ?= 1
PATIENCE_SET ?= 10
EPOCHS_SET ?= 60
IL_K_SET ?= 12
PICK_LOSS_SET ?= 0
VARIANT_SET ?= 0
LNG_SET ?= ady arm bul dut fre geo gre hin hun ice jpn kor lit rum vie 
NFK ?=   . .nfd.  
DYNET_SEED ?=  4 5 6
doit: 
	echo $(info training-done-files $(training-done-files))



# Template fuer das Kopieren vom build-Ordner in den Distributionsordner
define VLNBR_TMPL

training-done-files += $(RESULTSDIR)/m+$(input)+$(feat_input)+$(action_input)+$(enc_hidden)+$(dec_hidden)+$(enc_layers)+$(dec_layers)-a$(sed_aligner_em_iterations)+$(sed_aligner_discount)-t+$(dropout)+$(optimization)+$(batch_size)+$(patience)+$(epochs)+$(il_k)+$(pick_loss)-x$(variant)/$(lng)$(nfk)/s$(dynet_seed)/training.done


$(RESULTSDIR)/m+$(input)+$(feat_input)+$(action_input)+$(enc_hidden)+$(dec_hidden)+$(enc_layers)+$(dec_layers)-a$(sed_aligner_em_iterations)+$(sed_aligner_discount)-t+$(dropout)+$(optimization)+$(batch_size)+$(patience)+$(epochs)+$(il_k)+$(pick_loss)-x$(variant)/$(lng)$(nfk)/s$(dynet_seed)/training.done : 
	# create $$@
	INPUT=$(input) \
 FEAT_INPUT=$(feat_input)\
 ACTION_INPUT=$(action_input)\
 ENC_HIDDEN=$(enc_hidden)\
 DEC_HIDDEN=$(dec_hidden)\
 ENC_LAYERS=$(enc_layers)\
 DEC_LAYERS=$(dec_layers)\
 SED_ALIGNER_EM_ITERATIONS=$(sed_aligner_em_iterations)\
 SED_ALIGNER_DISCOUNT=$(sed_aligner_discount)\
 DROPOUT=$(dropout)\
 OPTIMIZATION=$(optimization)\
 BATCH_SIZE=$(batch_size)\
 PATIENCE=$(patience)\
 IL_K=$(il_k)\
 PICK_LOSS=$(pick_loss)\
 VARIANT=$(variant)\
 LNG=$(lng)\
 NFK=$(nfk)\
 DYNET_SEED=$(dynet_seed)\
 bash tests/sub_lang.bash && touch $$@
 
endef


#
$(eval \
 $(foreach input,$(INPUT_SET),\
  $(foreach feat_input,$(FEAT_INPUT_SET),\
   $(foreach action_input,$(ACTION_INPUT_SET),\
    $(foreach enc_hidden,$(ENC_HIDDEN_SET),\
     $(foreach dec_hidden,$(DEC_HIDDEN_SET),\
      $(foreach enc_layers,$(ENC_LAYERS_SET),\
       $(foreach dec_layers,$(DEC_LAYERS_SET),\
        $(foreach sed_aligner_em_iterations,$(SED_ALIGNER_EM_ITERATIONS_SET),\
         $(foreach sed_aligner_discount,$(SED_ALIGNER_DISCOUNT_SET),\
          $(foreach dropout,$(DROPOUT_SET),\
           $(foreach optimization,$(OPTIMIZATION_SET),\
            $(foreach batch_size,$(BATCH_SIZE_SET),\
             $(foreach patience,$(PATIENCE_SET),\
              $(foreach epochs,$(EPOCHS_SET),\
              $(foreach il_k,$(IL_K_SET),\
              $(foreach pick_loss,$(PICK_LOSS_SET),\
               $(foreach variant,$(VARIANT_SET),\
                $(foreach lng,$(LNG_SET),\
                 $(foreach nfk,$(NFK),\
                  $(foreach dynet_seed,$(DYNET_SEED),\
$(VLNBR_TMPL))))))))))))))))))))))



task: $(training-done-files)
