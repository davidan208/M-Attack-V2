#!/bin/bash

# Set NumExpr thread limit to suppress warnings
export NUMEXPR_MAX_THREADS=32

echo "Generating adversarial samples"

python generate_ad_sample_parallel.py "$@"

echo "Generating blackbox text"

python blackbox_text_generation.py "$@"

echo "Evaluating blackbox text"

python gpt_evaluate.py "$@" 

echo "Evaluating keywords matching"

python keyword_matching_gpt.py "$@"

