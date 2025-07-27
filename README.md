# XEmoGPT

## 🧩 Environment Setup

**Step 1: Create Conda Environment**

```bash
conda env create -f environment.yaml
conda activate XEmoGPT
````

**Step 2: Download Models and Set Paths**

* **Qwen3-4B**

  * Config file: `eval_configs/inference_config.yaml`
  * Link: [https://huggingface.co/Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)

* **chinese-hubert-large**

  * Config file: `inference.py`
  * Link: [https://huggingface.co/TencentGameMate/chinese-hubert-large](https://huggingface.co/TencentGameMate/chinese-hubert-large)

* **LaBSE**

  * Config file: `inference.py`
  * Link: [https://huggingface.co/sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE)

* **clip-vit-large-patch14**

  * Config file: `inference.py`
  * Link: [https://huggingface.co/openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

**Step 3: Download Checkpoint**

* Config file: `eval_configs/inference_config.yaml`

  * Link: [https://pan.baidu.com/s/1k7qFDCexMVdmKUKmLmxLTg?pwd=v5n2](https://pan.baidu.com/s/1k7qFDCexMVdmKUKmLmxLTg?pwd=v5n2)


---

## 🎬 How to Run Inference

Run the following command in your terminal, specifying paths to the input video, audio, and subtitle file:

```bash
python inference.py \
  --cfg-path eval_configs/inference_config.yaml \
  --video-path [your video path] \
  --audio-path [your audio path] \
  --subtitle [your subtitle file]
```

---

## 📐 EmoCue-360 Evaluation

### Download Required Model

* **all-MiniLM-L6-v2**

  * Config file: `EmoCue-360/compute_prf.py`
  * Link: [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

### Usage Instructions

#### Step 1: Extract Emotional Cues (`extract_clue.py`)

```python
INPUT_PATH = "/your/path/to/final-EMER-reason.csv"
emer_dict = pd.read_csv(INPUT_PATH).set_index('name')['english'].to_dict()
result = extract_vae_clue_from_dict(emer_dict)

OUTPUT_PATH = "/your/path/to/output.json"
save_dict_to_json(result, OUTPUT_PATH)
```

#### Step 2: Compute PRF Scores (`compute_prf.py`)

```python
MODEL_PATH = "/your/path/to/all-MiniLM-L6-v2"
model = EmoCue360(MODEL_PATH)

emer_gt_dict = load_json_to_dict("/your/path/to/EMER_clue.json")
rf_dict_xemogpt_emer = load_json_to_dict("/your/path/to/xemogpt_clue.json")

print("Visual: ", model.compute_visual_prf(rf_dict_xemogpt_emer, emer_gt_dict))
print("Auditory: ", model.compute_auditory_prf(rf_dict_xemogpt_emer, emer_gt_dict))
print("Global: ", model.compute_emotional_prf(rf_dict_xemogpt_emer, emer_gt_dict))
```

---

## 📂 EmoCue Dataset Download

* **EmoCue-Instruct**

  * Based on MER-Caption+ with refined annotations. Only annotation files are provided; video/audio files refer to [MER2025](https://huggingface.co/datasets/MERChallenge/MER2025)
  * Link: [https://pan.baidu.com/s/1JZ7CbXnsfEUMW0jwdEXzUA?pwd=9rh2](https://pan.baidu.com/s/1JZ7CbXnsfEUMW0jwdEXzUA?pwd=9rh2)

* **EmoCue-ShortCaption**

  * Based on [DFEW](https://dfew-dataset.github.io/download.html) and MER2025. Only annotation files are provided.
  * Link: [https://pan.baidu.com/s/1PyQ\_LqfOM9\_ZpRBWSna6Hw?pwd=4xi7](https://pan.baidu.com/s/1PyQ_LqfOM9_ZpRBWSna6Hw?pwd=4xi7)

* **EmoCue-Eval**

  * Evaluation benchmark based on MER2025. Only annotation files are provided.
  * Link: [https://pan.baidu.com/s/1hf7G\_DkttPmqVaA6Vtu8\_Q?pwd=6d75](https://pan.baidu.com/s/1hf7G_DkttPmqVaA6Vtu8_Q?pwd=6d75)
