import json
import numpy as np
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer

class EmoCue360():
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)

    def compute_recall_precision(self, rfs, gts):
        if len(rfs) == 0:
            return 0.0, 0.0
        
        assert len(gts) > 0

        # 批量编码
        gt_emb = self.model.encode(gts)
        rf_emb = self.model.encode(rfs)
        
        # 计算相似度矩阵
        sim_matrix = self.model.similarity(gt_emb, rf_emb)

        recalls, _ = torch.max(sim_matrix, dim=1)
        precisions, _ = torch.max(sim_matrix, dim=0)

        return recalls.mean().item(), precisions.mean().item()

    def compute_visual_prf(self, rf_dict, gt_dict):
        visual_recalls, visual_precisions, visual_f1s = [], [], []
        for k, v in tqdm(rf_dict.items(), desc="Compute_Visual_PRF"):
            visual_rfs, visual_gts = v["visual"], gt_dict[k]["visual"]
            if len(visual_gts) == 0:
                continue
            visual_recall, visual_precision = self.compute_recall_precision(visual_rfs, visual_gts)
            if visual_recall + visual_precision > 0:
                visual_f1 = 2 * visual_recall * visual_precision / (visual_precision + visual_recall)
            else:
                visual_f1 = 0

            visual_recalls.append(visual_recall)
            visual_precisions.append(visual_precision)
            visual_f1s.append(visual_f1)
        
        return {
            "recall": sum(visual_recalls)/len(visual_recalls),
            "precision": sum(visual_precisions)/len(visual_precisions),
            "f1": sum(visual_f1s)/len(visual_f1s)
        }

    def compute_auditory_prf(self, rf_dict, gt_dict):
        auditory_recalls, auditory_precisions, auditory_f1s = [], [], []
        for k, v in tqdm(rf_dict.items(), desc="Compute_Auditory_PRF"):
            auditory_rfs, auditory_gts = v["auditory"], gt_dict[k]["auditory"]
            flitered_auditory_rfs, filtered_auditory_gts = [], []

            # 由于字幕已经输入给了LLM，因此如果回答时计算字幕的分数是没有意义的，相当于作弊
            # 这里删除rf_dict和gt_dict中的字幕线索
            for auditory_rf in auditory_rfs:
                if "text" in auditory_rf:
                    continue
                flitered_auditory_rfs.append(auditory_rf)
            for auditory_gt in auditory_gts:
                if "text" in auditory_gt:
                    continue
                filtered_auditory_gts.append(auditory_gt)
            if len(filtered_auditory_gts) == 0:
                continue

            auditory_recall, auditory_precision = self.compute_recall_precision(flitered_auditory_rfs, filtered_auditory_gts)
            if auditory_recall + auditory_precision > 0:
                auditory_f1 = 2 * auditory_recall * auditory_precision / (auditory_recall + auditory_precision)
            else:
                auditory_f1 = 0

            auditory_recalls.append(auditory_recall)
            auditory_precisions.append(auditory_precision)
            auditory_f1s.append(auditory_f1)

        return {
            "recall": sum(auditory_recalls)/len(auditory_recalls),
            "precision": sum(auditory_precisions)/len(auditory_precisions),
            "f1": sum(auditory_f1s)/len(auditory_f1s)
        }
    
    def compute_emotional_prf(self, rf_dict, gt_dict):
        emotional_recalls, emotional_precisions, emotional_f1s = [], [], []
        for k, v in tqdm(rf_dict.items(), desc="Compute_Emotional_PRF"):
            if "emotional" not in v:
                emotional_recalls.append(0)
                emotional_precisions.append(0)
                emotional_f1s.append(0)
                continue
            emotional_rfs, emotional_gts = v["emotional"], gt_dict[k]["emotional"]
            if len(emotional_gts) == 0:
                continue

            emotional_recall, emotional_precision = self.compute_recall_precision(emotional_rfs, emotional_gts)
            # 防止分母为零
            if emotional_recall + emotional_precision > 0:
                emotional_f1 = 2 * emotional_recall * emotional_precision / (emotional_recall + emotional_precision)
            else:
                emotional_f1 = 0

            emotional_recalls.append(emotional_recall)
            emotional_precisions.append(emotional_precision)
            emotional_f1s.append(emotional_f1)
        
        return {
            "recall": sum(emotional_recalls)/len(emotional_recalls),
            "precision": sum(emotional_precisions)/len(emotional_precisions),
            "f1": sum(emotional_f1s)/len(emotional_f1s)
        }

def load_json_to_dict(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None


if __name__ == "__main__":
    MODEL_PATH = "/your/path/to/all-MiniLM-L6-v2"
    model = EmoCue360(MODEL_PATH)

    emer_gt_dict = load_json_to_dict("/your/path/to/EMER.json")
    rf_dict_xemogpt_emer = load_json_to_dict("/your/path/to/XEmoGPT.json")
    print("Visual: ", model.compute_visual_prf(rf_dict_xemogpt_emer, emer_gt_dict))
    print("Auditory: ", model.compute_auditory_prf(rf_dict_xemogpt_emer, emer_gt_dict))
    print("Emotional: ", model.compute_emotional_prf(rf_dict_xemogpt_emer, emer_gt_dict))
