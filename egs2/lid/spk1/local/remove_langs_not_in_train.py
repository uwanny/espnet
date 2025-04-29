
train_spk2utt_dir = "/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/dump/raw/train_ml_superb2_lang/spk2utt"
train_langs = set()

with open(train_spk2utt_dir, "r") as f:
    for line in f:
        lang = line[:3]
        train_langs.add(lang)

dev_utt2spk_dir = "/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/dump/raw/dev_ml_superb2_lang/utt2spk"
dev_utt2spk_after_remove_dir = "/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/dump/raw/dev_ml_superb2_lang/utt2spk_after_remove_lang_in_train_not_in_dev"
dev_wavscp_dir = "/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/dump/raw/dev_ml_superb2_lang/wav.scp"
dev_wavscp_after_remove_dir = "/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/dump/raw/dev_ml_superb2_lang/wav.scp_after_remove_lang_in_train_not_in_dev"

remove_ids = set()

dev_utt2spk_after_remove_dump = []
with open(dev_utt2spk_dir, "r") as f:
    for line in f:
        utt_id, lang = line.strip().split()
        if lang not in train_langs:
            remove_ids.add(utt_id)
        else:
            dev_utt2spk_after_remove_dump.append(line.strip())
with open(dev_utt2spk_after_remove_dir, "w") as f:
    for line in dev_utt2spk_after_remove_dump:
        f.write(line + "\n")

dev_wavscp_after_remove_dump = []
with open(dev_wavscp_dir, "r") as f:
    for line in f:
        utt_id = line.strip().split()[0]
        if utt_id not in remove_ids:
            dev_wavscp_after_remove_dump.append(line.strip())
with open(dev_wavscp_after_remove_dir, "w") as f:
    for line in dev_wavscp_after_remove_dump:
        f.write(line + "\n")
        