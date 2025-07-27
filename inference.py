import os
import json
import yaml
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import lightning.pytorch as pl

from model.audio_encoder import HuBERTEncoder, AudioEncoder, AudioTextEncoder
from model.visual_encoder import CLIPVisualEncoder, VisualTextEncoder
from model.xemogpt import XEmoGPT
from utils.video_processor import load_audio, load_video

def load_json_to_dict(path):
    with open(path, 'r') as f:
        return json.load(f)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-path', type=str)
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--audio-path', type=str)
    parser.add_argument('--subtitle', type=str)
    return parser.parse_args()

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]

if __name__ == "__main__":
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    args = parse_config()
    model_cfg = load_cfg(args.cfg_path)

    # HuBERT+ATEB
    audiotext_encoder = AudioTextEncoder(
        audio_encoder_path="/your/path/to/chinese-hubert-large",
        text_encoder_path="/your/path/to/LaBSE",
        num_alignment_layers=2
    )
    audio_encoder = audiotext_encoder.audio_encoder
    # CLIP+VTEB
    visualtext_encoder = VisualTextEncoder(
        visual_encoder_path="/your/path/to/clip-vit-large-patch14",
        num_alignment_layers=2,
        use_attention_pooling=True,
        use_temporal_contrastive=True,
        use_masked_frame_modeling=True
    )
    visual_encoder = visualtext_encoder
    # XEmoGPT
    model = XEmoGPT.from_config(model_cfg, visual_encoder, audio_encoder).to("cuda:1")

    # Load Video and Audio
    audio = load_audio(args.audio_path)
    video_frames = load_video(args.video_path)
    sentence = args.subtitle

    reason_instruction_pool = """
                Please describe the speaker's emotional state in the video. 
                **Output Requirements:**  
                Present the synthesized multimodal information in a well-structured paragraph, such as: 'In the video, ... In the audio, ...'. 
                Conclude the paragraph with an emotion state inference based on all cues. 
                """
    instruction = f"###Human: The audio content is as follows: <Audio><AudioHere></Audio>. " \
                + f"Meanwhile, we uniformly sample raw frames from the video: <Video><VideoHere></Video>. "  \
                + f"The subtitle of this video is: <Subtitle>{sentence}</Subtitle>. " \
                + f"Now, please answer my question based on all the provided information. {reason_instruction_pool} ###Assistant: "

    response = model.generate([video_frames], [audio], [instruction])
    print(response[0])


    