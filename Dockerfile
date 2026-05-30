FROM vllm/vllm-omni:latest-aarch64

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git libsndfile1 ffmpeg nano vim tree \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    aenum diffusers librosa soundfile soxr

# Clone repo
RUN git clone https://github.com/davidan208/M-Attack-V2.git /workspace/M-Attack-V2

# Install project dependencies
WORKDIR /workspace/M-Attack-V2
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    hydra-core \
    wandb 

CMD ["/bin/bash"]
