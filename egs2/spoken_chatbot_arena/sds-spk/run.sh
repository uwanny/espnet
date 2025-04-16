#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

# export HF_TOKEN=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


[ -z "${HF_TOKEN}" ] && \
    echo "ERROR: You need to setup the variable HF_TOKEN with your HuggingFace access token" && \
exit 1

source /ocean/projects/cis210027p/jsunc/espnet/tools/miniconda/etc/profile.d/conda.sh
conda activate espnet

export TRANSFORMERS_CACHE=/ocean/projects/cis210027p/jsunc/espnet/egs2/spoken_chatbot_arena/sds-spk
export HF_HOME=/ocean/projects/cis210027p/jsunc/espnet/egs2/spoken_chatbot_arena/sds-spk
export HF_DATASETS_CACHE=/ocean/projects/cis210027p/jsunc/espnet/egs2/spoken_chatbot_arena/sds-spk
export TORCHAUDIO_CACHE_DIR=/ocean/projects/cis210027p/jsunc/espnet/egs2/spoken_chatbot_arena/sds-spk
export NLTK_DATA=/ocean/projects/cis210027p/jsunc/espnet/egs2/spoken_chatbot_arena/sds-spk
export GDOWN_CACHE_DIR=/ocean/projects/cis210027p/jsunc/espnet/egs2/spoken_chatbot_arena/sds-spk
export GRADIO_CACHE_DIR=/ocean/projects/cis210027p/jsunc/espnet/egs2/spoken_chatbot_arena/sds-spk
export TMPDIR=/ocean/projects/cis210027p/jsunc/espnet/egs2/spoken_chatbot_arena/sds-spk





python app.py \
  --eval_options "Latency,TTS Intelligibility,TTS Speech Quality,ASR WER,Text Dialog Metrics" \
  --tts_options "kan-bayashi/ljspeech_vits,kan-bayashi/libritts_xvector_vits,kan-bayashi/vctk_multi_spk_vits,ChatTTS" \
  --llm_options "meta-llama/Llama-3.2-1B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct" \
  --asr_options "pyf98/owsm_ctc_v3.1_1B,espnet/owsm_ctc_v3.2_ft_1B,espnet/owsm_v3.1_ebf,librispeech_asr,whisper-large" \
  --spk_options "espnet/voxcelebs12_rawnet3"


# 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



