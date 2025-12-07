
import torch
import argparse
from moshi.models import loaders
from moshi.models.lm import LMGen
from moshi.conditioners.base import ConditionType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading Moshi model...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    model = loaders.get_moshi_lm(
        checkpoint_info.moshi_weights,
        lm_kwargs=checkpoint_info.lm_config,
        device=args.device,
        dtype=torch.bfloat16
    )
    model.eval()

    print(f"Model loaded.")
    print(f"Condition Provider: {model.condition_provider}")
    print(f"Fuser: {model.fuser}")
    
    if model.fuser:
        print("Fuser configuration:", model.fuser.fuse2cond)
    else:
        print("No fuser found. Model does not support conditioning.")

if __name__ == "__main__":
    main()
