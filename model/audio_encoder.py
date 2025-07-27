import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2FeatureExtractor, HubertModel
from transformers.models.hubert.modeling_hubert import HubertEncoderLayer
from sentence_transformers import SentenceTransformer

class AudioEncoder(nn.Module):
    def __init__(self, audio_encoder_path, num_alignment_layers, output_dim=768):
        super(AudioEncoder, self).__init__()
        self.audio_encoder = HubertModel.from_pretrained(audio_encoder_path)
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_path)
        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        self.audio_encoder = self.audio_encoder.eval()
        self.config = self.audio_encoder.config

        # ATEB
        self.audio_alignment_layer = nn.ModuleList([HubertEncoderLayer(self.audio_encoder.config) for _ in range(num_alignment_layers)])
        self.audio_project = nn.Linear(self.audio_encoder.config.hidden_size, output_dim)
    
    def forward(self, audios):
        inputs = self.audio_processor(audios, padding=True, return_tensors="pt",sampling_rate=16000).to(self.get_device())
        with torch.no_grad():
            audio_feature = self.audio_encoder(**inputs, return_dict=True).last_hidden_state
        
        ### this code from transformers/models/hubert/modeling_hubert.py
        attention_mask = self.audio_encoder._get_feature_vector_attention_mask(audio_feature.shape[1], inputs["attention_mask"])
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, audio_feature.shape[2])
        audio_feature[~expand_attention_mask] = 0.0

        ### this code from transformers/models/hubert/modeling_hubert.py
        full_attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=audio_feature.dtype)
        full_attention_mask = full_attention_mask * torch.finfo(audio_feature.dtype).min
        full_attention_mask = full_attention_mask.expand(
            full_attention_mask.shape[0], 1, full_attention_mask.shape[-1], full_attention_mask.shape[-1]
        )

        for layer in self.audio_alignment_layer:
            layer_outputs = layer(
                audio_feature, attention_mask=full_attention_mask
            )
            audio_feature = layer_outputs[0]
        
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, audio_feature.shape[2])
        audio_feature[~expand_attention_mask] = 0.0
        pooled_output = audio_feature.sum(dim=1) / attention_mask.sum(dim=1).view(-1, 1)
        pooled_output = self.audio_project(pooled_output)

        return {
            "audio_feature": audio_feature,
            "pooled_output": pooled_output,
            "attention_mask": attention_mask,
        }
    
    def get_device(self):
        return next(self.parameters()).device


class TextEncoder(nn.Module):
    def __init__(self, text_encoder_path, output_dim=768):
        super(TextEncoder, self).__init__()
        self.text_encoder = SentenceTransformer(text_encoder_path)
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        self.text_encoder = self.text_encoder.eval()

        self.text_project = nn.Linear(self.text_encoder.get_sentence_embedding_dimension(), output_dim)

    def forward(self, texts):
        with torch.no_grad():
            pooled_output = self.text_encoder.encode(texts, convert_to_tensor=True)
        pooled_output = self.text_project(pooled_output)
        return pooled_output
    
    def get_device(self):
        return next(self.parameters()).device


class AudioTextEncoder(nn.Module):
    def __init__(self, audio_encoder_path, text_encoder_path, num_alignment_layers=1, output_dim=768):
        super().__init__()
        self.audio_encoder = AudioEncoder(audio_encoder_path, num_alignment_layers, output_dim)
        self.text_encoder = TextEncoder(text_encoder_path, output_dim)

    def forward(self, batch):
        audios, annotations = batch['audios'], batch['annotations']
        
        audio_features = self.audio_encoder(audios)["pooled_output"]  # [local_batch, dim]
        text_features = self.text_encoder(annotations)               # [local_batch, dim]
        
        return audio_features, text_features
    


class HuBERTEncoder(nn.Module):
    def __init__(self, audio_encoder_path):
        super(HuBERTEncoder, self).__init__()
        self.audio_encoder = HubertModel.from_pretrained(audio_encoder_path)
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_encoder_path)
        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False
        self.audio_encoder = self.audio_encoder.eval()
        self.config = self.audio_encoder.config
    
    def forward(self, audios):
        # audios: List(np.ndarray) or List(torch.Tensor)
        inputs = self.audio_processor(audios, padding=True, return_tensors="pt",sampling_rate=16000).to(self.get_device())
        with torch.no_grad():
            audio_feature = self.audio_encoder(**inputs, return_dict=True).last_hidden_state
        
            attention_mask = self.audio_encoder._get_feature_vector_attention_mask(audio_feature.shape[1], inputs["attention_mask"])
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, audio_feature.shape[2])
            audio_feature[~expand_attention_mask] = 0.0

        return {
            "audio_feature": audio_feature,
            "attention_mask": attention_mask,
        }
    
    def get_device(self):
        return next(self.parameters()).device