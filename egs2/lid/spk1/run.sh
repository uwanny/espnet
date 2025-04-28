#!/usr/bin/env bash
set -e
set -u
set -o pipefail

export HF_HOME=/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1


train_set="train_fleurs_lang"
valid_set="dev_fleurs_lang"

# in domain test
# test_sets="test_fleurs_lang"

# out of domain test
test_sets="test_voxlingua107_lang dev_ml_superb2_lang dev_ml_superb2_dialect_lang"

feats_type="raw"
inference_model="valid.loss.best.pth"

./lid_fix_inference.sh \
    --feats_type ${feats_type} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_model ${inference_model} \
    --inference_batch_size 8 \
    --extract_embd false \
    --save_every 1000 \
    --nj 8 \
    --ngpu 1 \
    --spk_args "
        --use_wandb true 
        --wandb_project lid 
        --wandb_entity qingzhew-carnegie-mellon-university
        " \
    "$@"
