from __future__ import annotations

import os

import random

import time

import psutil

import torch

from math_framework import (
    GPT,
    BLOCK_SIZE,
    BATCH_SIZE,
    VOCAB_SIZE_EXPECTED,
    count_parameters,
    create_eval_batches,
    evaluate_fixed,
    get_batch,
    load_dataset,
)

from optimizer import (
    AttentionMLPWheelOptimizer,
)

SEED = 1337

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required for this 100M parameter benchmark."
    )

DEVICE = torch.device("cuda")

D_MODEL = 1024

N_LAYER = 8

N_HEAD = 16

MAX_STEPS = 1500

BASE_INFINITESIMAL = 2e-3

WARMDOWN_START = 1000

FINAL_INFINITESIMAL = 2e-4

PROJECTIVE_CEILING = 6.0

WEIGHT_DECAY = 1e-2

ALPHA = 0.19

LOG_INTERVAL_SECONDS = 30.0

EVAL_INTERVAL = 100

GENERATE_TOKENS = 500

TEMPERATURE = 0.85

TOP_K = 40

CHECKPOINT_FILE = (
    "wheelsgd_tinyshakespeare_100m_best.pt"
)

random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )

try:

    torch.set_num_threads(
        max(
            1,
            os.cpu_count() or 1,
        )
    )

except Exception: