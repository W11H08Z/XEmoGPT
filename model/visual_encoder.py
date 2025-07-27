import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, CLIPImageProcessorFast, CLIPTokenizerFast
from transformers.models.clip.modeling_clip import CLIPEncoderLayer

class CLIPVisualEncoder(nn.Module):
    def __init__(self, visual_encoder_path):
        super(CLIPVisualEncoder, self).__init__()
        self.visual_encoder = CLIPModel.from_pretrained(visual_encoder_path)
        self.visual_imageprocessor = CLIPImageProcessorFast.from_pretrained(visual_encoder_path)
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.config = self.visual_encoder.config

    def forward(self, images):
        # images: List[PIL.Image]
        inputs = self.visual_imageprocessor(images=images, return_tensors="pt").to(self.get_device())
        with torch.no_grad():
            visual_feature = self.visual_encoder.get_image_features(**inputs)
        return visual_feature

    def get_visual_feature_from_frame(self, images):
        return self.forward(images)

    def get_device(self):
        return next(self.parameters()).device



# ---------- Attention Pooling ------------
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_vector = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        # x: (batch, time, dim)
        weights = torch.softmax(x @ self.attn_vector, dim=1)  # (batch, time)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=1)
        return pooled

# ---------- VisualTextEncoder ------------
class VisualTextEncoder(nn.Module):
    def __init__(
        self,
        visual_encoder_path,
        num_alignment_layers,
        output_dim=768,
        use_attention_pooling=False,
        use_temporal_contrastive=False,
        use_masked_frame_modeling=False
    ):
        super().__init__()

        self.num_frames = 16

        # Load frozen CLIP
        self.clip = CLIPModel.from_pretrained(visual_encoder_path)
        self.clip_imageprocessor = CLIPImageProcessorFast.from_pretrained(visual_encoder_path)
        self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(visual_encoder_path)
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.eval()

        self.config = self.clip.config

        # VTEB
        self.visual_alignment_layer = nn.ModuleList([
            CLIPEncoderLayer(self.clip.config.vision_config)
            for _ in range(num_alignment_layers)
        ])

        self.visual_project = nn.Linear(self.clip.config.vision_config.hidden_size, output_dim)
        self.text_project = nn.Linear(self.clip.config.text_config.hidden_size, output_dim)

        # Positional Encoding
        self.position_embedding = nn.Parameter(
            torch.zeros(1, 16, self.clip.config.vision_config.hidden_size)
        )
        nn.init.normal_(self.position_embedding, std=0.02)

        # Flags
        self.use_attention_pooling = use_attention_pooling
        self.use_temporal_contrastive = use_temporal_contrastive
        self.use_masked_frame_modeling = use_masked_frame_modeling

        # Optional modules
        if self.use_attention_pooling:
            self.attention_pooling = AttentionPooling(self.clip.config.vision_config.hidden_size)

        if self.use_temporal_contrastive:
            self.temporal_disc_head = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 1)
            )

        if self.use_masked_frame_modeling:
            self.frame_reconstructor = nn.Linear(
                self.clip.config.vision_config.hidden_size,
                self.clip.config.vision_config.hidden_size
            )

    def get_device(self):
        return next(self.parameters()).device

    def extract_frame_features(self, pixel_values):
        bsz, len_frames = len(pixel_values)//self.num_frames, self.num_frames
        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        visual_feats = vision_outputs.last_hidden_state
        visual_feats = visual_feats.view(bsz, len_frames, -1, self.clip.config.vision_config.hidden_size)[:, :, 0, :]
        return visual_feats

    def forward(self, batch, compute_shuffled=False, compute_reconstruction=False):
        pixel_values, annotations = batch['pixel_values'], batch['annotations']
        bsz, len_frames = len(pixel_values)//self.num_frames, self.num_frames

        # 1. 先提取原始完整帧特征 (缓存)
        original_feats = self.extract_frame_features(pixel_values)  # [bsz, T, feat_dim]

        # 2. 带mask特征 (用于MFM)
        mask_info = None
        visual_feats_masked = original_feats
        if self.use_masked_frame_modeling and self.training:
            mask_ratio = 0.5
            mask = (torch.rand(original_feats.shape[:2], device=original_feats.device) < mask_ratio)
            visual_feats_masked = original_feats.masked_fill(mask.unsqueeze(-1), 0)
            mask_info = mask

        # 3. 给所有视觉特征加位置编码
        original_feats = original_feats + self.position_embedding[:, :len_frames, :]
        visual_feats_masked = visual_feats_masked + self.position_embedding[:, :len_frames, :]

        # 4. 对齐层处理完整特征（主对齐用）
        aligned_feats = original_feats
        for layer in self.visual_alignment_layer:
            aligned_feats = layer(aligned_feats, None, None)[0]

        # 5. 池化得到视觉向量
        if self.use_attention_pooling:
            pooled_visual = self.attention_pooling(aligned_feats)
        else:
            pooled_visual = aligned_feats.mean(dim=1)
        visual_emb = self.visual_project(pooled_visual)

        # 6. 文本编码器编码文本（不训练）
        text_inputs = self.clip_tokenizer(
            text=annotations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.get_device())
        with torch.no_grad():
            text_feats = self.clip.get_text_features(**text_inputs)
        text_emb = self.text_project(text_feats)

        # 7. 对齐层处理带mask特征（用于MFM重建）
        aligned_feats_masked = visual_feats_masked
        for layer in self.visual_alignment_layer:
            aligned_feats_masked = layer(aligned_feats_masked, None, None)[0]

        # 8. 乱序特征（时序对比用）
        shuffled_emb = None
        if compute_shuffled and self.use_temporal_contrastive:
            idx = torch.randperm(len_frames)
            shuffled_feats = original_feats[:, idx, :]
            for layer in self.visual_alignment_layer:
                shuffled_feats = layer(shuffled_feats, None, None)[0]
            if self.use_attention_pooling:
                pooled_shuffled = self.attention_pooling(shuffled_feats)
            else:
                pooled_shuffled = shuffled_feats.mean(dim=1)
            shuffled_emb = self.visual_project(pooled_shuffled)

        # 9. MFM重建预测
        pred_recon = None
        if compute_reconstruction and self.use_masked_frame_modeling:
            pred_recon = self.frame_reconstructor(aligned_feats_masked)

        return visual_emb, text_emb, mask_info, aligned_feats_masked, shuffled_emb, pred_recon, original_feats

    @torch.no_grad()
    def get_original_frame_features(self, batch):
        pixel_values = batch['pixel_values']
        return self.extract_frame_features(pixel_values)
    
    
    def get_visual_feature_from_frame(self, frames):
        bsz, len_frames = len(frames), len(frames[0])
        pixel_values = self.clip_imageprocessor(images=frames, return_tensors="pt")["pixel_values"].to(self.get_device())
        original_feats = self.extract_frame_features(pixel_values)
        original_feats = original_feats + self.position_embedding[:, :len_frames, :]

        aligned_feats = original_feats
        for layer in self.visual_alignment_layer:
            aligned_feats = layer(aligned_feats, None, None)[0]

        return aligned_feats