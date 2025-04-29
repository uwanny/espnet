import json

with open("/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/exp/spk_train_lid_mms_unfrozen_xvector_origin_lr5e-6_bs6m_step30k_3warm_fix_catpow_non_fix_duration_max_bs8_raw/inference/test_fleurs_lang_lids", "r") as f:
    lid_dict = json.load(f)

with open("/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/exp/spk_train_lid_mms_unfrozen_xvector_origin_lr5e-6_bs6m_step30k_3warm_fix_catpow_non_fix_duration_max_bs8_raw/inference/test_fleurs_lang_lids", "w") as f:
    for utt, lid in lid_dict.items():
        f.write(f"{utt} {lid}\n")