#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_shift=300

acoustic_model="english_us_arpa"
./scripts/utils/mfa.sh \
    --split_sets "train-clean-100 dev-clean test-clean" \
    --language english_us_espeak  \
    --acoustic_model ${acoustic_model} \
    --dictionary ${acoustic_model} \
    --cleaner tacotron \
    --samplerate ${fs} \
    --hop-size ${n_shift} \
    --clean_temp true \
    "$@"

    # --train true \
    # --g2p_model espeak_ng_english_us_vits \
