import os
from PIL import Image
import numpy as np

import librosa
import decord
from decord import VideoReader, cpu, gpu

import torch


def load_video(video_path, num_frames=16):
    """使用 decord 高效读取 MP4 的所有帧，并等间隔选取指定帧数
    
    Args:
        video_path (str): 视频路径
        use_gpu (bool): 是否使用 GPU 加速（需要 CUDA）
        num_frames (int): 要选取的帧数（默认 8 帧）
    
    Returns:
        List[Image.Image]: 等间隔选取的 PIL.Image 列表
    """
    device = cpu()
    vr = VideoReader(video_path, ctx=device)
    
    # 计算均匀分布的帧索引
    total_frames = len(vr)
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    # 提取指定帧
    frames = vr.get_batch(frame_indices).asnumpy()  # shape: (N, H, W, 3)
    
    # 转换为 PIL.Image
    pil_images = [Image.fromarray(frame) for frame in frames]
    
    return pil_images

def load_audio(audio_path, target_sample_rate=16000):
    # 使用librosa加载音频文件
    waveform, sample_rate = librosa.load(audio_path, 
                                        sr=target_sample_rate,  # 自动重采样到目标采样率
                                        mono=True,             # 强制转换为单声道
                                        dtype=np.float32)      # 使用float32格式
    
    max_length = 12 * 16000  # 12秒的采样点数(假设采样率是16000Hz)
    if len(waveform) > max_length:
        waveform = waveform[:max_length]  # 截取前12秒

    return waveform