#!/usr/bin/env python

import argparse
import torch
from safetensors.torch import save_file
from src.chatterbox.models.s3gen.s3gen import S3Token2Wav

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch checkpoint (.pt) of S3Token2Wav to safetensors (.safetensors)"
    )
    parser.add_argument(
        "--input_checkpoint",
        type=str,
        required=True,
        help="Path to the trained checkpoint file (.pt)"
    )
    parser.add_argument(
        "--output_safetensors",
        type=str,
        required=True,
        help="Output path for the .safetensors file"
    )
    args = parser.parse_args()

    # Load the PyTorch checkpoint
    print(f"🔄 Loading checkpoint from: {args.input_checkpoint}")
    ckpt = torch.load(args.input_checkpoint, map_location="cpu")

    # Instantiate the S3Token2Wav model
    model = S3Token2Wav()

    # Load the state_dict into the model (strict=False to tolerate missing/extra keys)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Save the model state in safetensors format
    print(f"💾 Saving model as safetensors to: {args.output_safetensors}")
    save_file(model.state_dict(), args.output_safetensors)

    print("✅ Model successfully saved as .safetensors")


if __name__ == "__main__":
    main()


# Example usage:
# python convert_to_safetensors.py \
#     --input_checkpoint /path/to/s3mel_epoch3_step52290.pt \
#     --output_safetensors /path/to/s3gen.safetensors
