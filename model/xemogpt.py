import time
import random

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
)


class XEmoGPT(nn.Module):
    def __init__(self, cfg, visual_encoder, audio_encoder):
        super().__init__()

        for key, value in cfg.items():
            try:
                setattr(self, key, value)
            except:
                print(f"Error setting attribute {key} with value {value}")

        # Qwen2.5-7B-Instruct, Qwen3-4B
        print("Loading LLM...")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
            device_map="cpu"
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
        if self.lora_r > 0:
            loraconfig = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm_model = get_peft_model(self.llm_model, loraconfig)
            self.llm_model.print_trainable_parameters()
        elif self.lora_r == 0:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
            print("Full LLM parameters are frozen.")
        elif self.lora_r < 0:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = True
            print("Full LLM parameters are trainable.")
        print("Loading LLM Done")

        # CLIP-VIT-Large-Patch14
        print("Loading Visual Encoder...")
        self.visual_encoder = visual_encoder
        self.visual_llm_project = nn.Linear(self.visual_encoder.config.vision_config.hidden_size, self.llm_model.config.hidden_size)
        print("Loading Visual Encoder Done")

        # HuBERT
        print("Loading Audio Encoder...")
        self.audio_encoder = audio_encoder
        self.audio_llm_project = nn.Linear(self.audio_encoder.config.hidden_size, self.llm_model.config.hidden_size)
        print("Loading Audio Encoder Done")

    def encode_image(self, frames):
        """
        frames: List [[PIL.Image*num_frames]*batch_size]
        """
        batch_size = len(frames)

        none_index = [index for index, value in enumerate(frames) if value is None]
        filtered_frames = [value for value in frames if value is not None]

        if len(filtered_frames) == 0:
            return [None] * batch_size

        num_frames = len(filtered_frames[0]) if batch_size > 0 else 0
        visual_feature = self.visual_encoder.get_visual_feature_from_frame(filtered_frames) # (batch_size*num_frames, feat_dim)
        visual_feature = self.visual_llm_project(visual_feature).reshape(batch_size-len(none_index), num_frames, -1)  # (batch_size, num_frames, llm_feat_dim)

        result_list = [None] * batch_size
        none_count = 0
        for i, value in enumerate(frames):
            if value is not None:
                result_list[i] = visual_feature[i - none_count]
            else:
                none_count += 1

        return result_list
    
    def encode_audio(self, audio):
        """
        audio: List [np.ndarray*batch_size]
        """
        batch_size = len(audio)
        none_index = [index for index, value in enumerate(audio) if value is None]
        filtered_audio = [value for value in audio if value is not None]

        if len(filtered_audio) == 0:
            return [None] * batch_size

        outputs = self.audio_encoder(filtered_audio)
        audio_feature, attention_mask = outputs["audio_feature"], outputs["attention_mask"]
        pooled_feature = audio_feature
        audio_feature = self.audio_llm_project(pooled_feature)

        result_list = [None] * batch_size
        none_count = 0
        for i, value in enumerate(audio):
            if value is not None:
                current_mask = attention_mask[i - none_count]
                result_list[i] = audio_feature[i - none_count][current_mask == 1]
            else:
                none_count += 1

        return result_list
    
    def prompt_wrap(self, img_embeds=None, aud_embeds=None, prompts=None):        
        emb_lists = []
        for idx, (each_prompt) in enumerate(prompts):
            interleave_emb, p_segs = [], [each_prompt]
            each_aud_embed = aud_embeds[idx]
            if each_aud_embed is not None:
                p_segs = p_segs[-1].split('<AudioHere>')
                p_tokens = self.llm_tokenizer(
                    p_segs[0], return_tensors="pt", add_special_tokens=False).to(each_aud_embed.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                interleave_emb.append(torch.cat([p_embed, each_aud_embed[None][:, :]], dim=1))
            
            each_img_embed = img_embeds[idx]
            if each_img_embed is not None:
                p_segs = p_segs[-1].split('<VideoHere>')
                p_tokens = self.llm_tokenizer(
                    p_segs[0], return_tensors="pt", add_special_tokens=False).to(each_img_embed.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, :]], dim=1))

            p_tokens = self.llm_tokenizer(
                    p_segs[-1], return_tensors="pt", add_special_tokens=False).to(self.get_device())
            p_embed = self.embed_tokens(p_tokens.input_ids)
            interleave_emb.append(p_embed)


            wrapped_emb = torch.cat(interleave_emb, dim=1)
            emb_lists.append(wrapped_emb)

        emb_lens = [emb.shape[1] for emb in emb_lists]
        pad_emb = self.embed_tokens(torch.tensor(self.llm_tokenizer.pad_token_id, device=self.get_device()))

        max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
        wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
        wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=self.get_device())
        
        for i, emb in enumerate(emb_lists):
            length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
            wrapped_embs[i, :length] = emb[:, :length]
            wrapped_atts[i, :length] = 1
        
        return wrapped_embs, wrapped_atts

    def preparing_embedding(self, samples):
        if "video_frames" in samples:
            img_embeds = self.encode_image(samples["video_frames"]) # img_embeds: tensor (batch_size, num_frames, feat_dim)
        else:
            img_embeds = None

        if "audios" in samples:
            aud_embeds = self.encode_audio(samples["audios"]) # aud_embeds: tensor (batch_size, num_select, feat_dim)
        else:
            aud_embeds = None

        # print(img_embeds.shape, aud_embeds.shape)

        instruction = samples["instructions"]
        cond_embeds, cond_atts = self.prompt_wrap(img_embeds=img_embeds, aud_embeds=aud_embeds, prompts=instruction)

        ### prepare target tokens
        text = [t + self.llm_tokenizer.eos_token for t in samples["annotations"]]

        regress_tokens = self.llm_tokenizer(
            text,
            return_tensors="pt",
            padding="longest"
        ).to(cond_embeds.device)

        regress_token_ids = regress_tokens.input_ids
        regress_atts = regress_tokens.attention_mask
        part_targets = regress_token_ids.masked_fill(
            regress_token_ids == self.llm_tokenizer.pad_token_id, -100
        )

        regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets
    
    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """

        input_lens = []
        cat_embs = []
        cat_atts = []

        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)

            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )

        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def forward(self, samples):
        # prepare the embedding to condition and the embedding to regress
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding(samples)

        # concat the embedding to condition and the embedding to regress
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)
        # get bos token embedding
        bos = torch.ones_like(part_targets[:, :1]) * self.llm_tokenizer.pad_token_id # in qwen2.5 pad_token is bos_token
        bos_embeds = self.embed_tokens(bos)
        bos_atts = attention_mask[:, :1]

        # add bos token at the begining
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(inputs_embeds.device).fill_(-100)
        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            reduction="mean",
        )
        loss = outputs.loss

        return loss

    @torch.no_grad()
    def generate(self, images, audios, texts, max_new_tokens=756, do_sample=True, num_beams=5, temperature=0.3, repetition_penalty=1.0, top_p=0.9):
        '''
            function for generate test use
        '''

        if images is not None:
            img_embeds = self.encode_image(images)
        else:
            img_embeds = None
        
        if audios is not None:
            aud_embeds = self.encode_audio(audios)
        else:
            aud_embeds = None

        embs, attn_mask = self.prompt_wrap(img_embeds, aud_embeds, texts)

        # get bos token embedding
        bos = torch.ones((len(texts), 1), dtype=torch.long).to(embs.device) * self.llm_tokenizer.pad_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = attn_mask[:, :1]

        # add bos token at the begining
        embs = torch.cat([bos_embeds, embs], dim=1)
        attn_mask = torch.cat([bos_atts, attn_mask], dim=1)

        outputs = self.llm_model.generate(
            inputs_embeds=embs,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p
        )

        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llm_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers

    def embed_tokens(self, token_ids):
        try:
            embeds = self.llm_model.base_model.model.model.embed_tokens(token_ids)
        except AttributeError:
            embeds = self.llm_model.model.embed_tokens(token_ids)

        return embeds

    def get_device(self):
        return next(self.parameters()).device

    @classmethod
    def from_config(cls, cfg, visual_encoder, audio_encoder):
        model = cls(
            cfg=cfg,
            visual_encoder=visual_encoder,
            audio_encoder=audio_encoder
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights
        if ckpt_path:
            print("Loading Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"], strict=False)  
        return model