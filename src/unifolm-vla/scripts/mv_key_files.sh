# Model files
cp src/unifolm-vla/model/framework/* projects/unifolm-vla/src/unifolm_vla/model/framework
cp src/unifolm-vla/model/modules/action_model/flow_matching_modules/* projects/unifolm-vla/src/unifolm_vla/model/modules/action_model/flow_matching_modules
cp src/unifolm-vla/model/modules/action_model/cross_attention_dit.py projects/unifolm-vla/src/unifolm_vla/model/modules/action_model/
cp src/unifolm-vla/model/modules/vlm/* projects/unifolm-vla/src/unifolm_vla/model/modules/vlm
# Evaluation script
cp src/unifolm-vla/LIBERO/eval_libero.py projects/unifolm-vla/experiments/LIBERO/