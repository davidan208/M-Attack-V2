#!/bin/bash

echo "Generating adversarial samples"

python generate_ad_sample_batch.py "$@"

echo "Generating blackbox text"

python blackbox_text_generation.py "$@"

echo "Evaluating blackbox text"

python gpt_evaluate.py "$@" 