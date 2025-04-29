# we save utt2lang as utt2spk, since it is the requirement of the Kaldi data preparation
dump_data_dir = "/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/dump/raw/dev_ml_superb2_dialect_lang"

utt2lang = open(f"{dump_data_dir}/utt2spk_lang", "w")

text_file = f"{dump_data_dir}/text"

with open(text_file, "r") as f:
    for line in f:
        utt, lang = line.strip().split(" ")
        utt2lang.write(utt + " " + lang + "\n")

utt2lang.close()