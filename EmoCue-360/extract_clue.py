import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import threading

client = OpenAI(
    api_key="xxxxxx",
    base_url="https://api.deepseek.com",
)

system_prompt1 = """
The user will provide a multimodal text describing the emotional state of a video character, including visual, auditory, and affective information. Parse this text into a structured JSON format, categorizing the information into three dimensions: visual, auditory, and emotional. Each dimension should be a list of concise, atomic statements that describe individual features or states.
"""
system_prompt2 = """
Analyze the user's multimodal description of a video character's emotional state, which includes visual, auditory, and affective cues. Transform this input into a structured JSON output with three distinct categories: visual, auditory, and emotional. Each category should contain a list of brief, individual observations that capture specific features or states.
"""
system_prompt3 = """
Your task is to process a multimodal description of a character's emotions in a video scene. The input contains visual, audio, and emotional information. Convert this into a well-organized JSON structure with three sections: visual (for appearance details), auditory (for sound-related elements), and emotional (for feelings and moods). Each section should list separate, concise observations.
"""
system_prompt4 = """
When provided with a detailed account of a video character's emotional state across multiple modalities (visual, auditory, affective), parse this information into a clean JSON structure. The output should have three main arrays: visual (for physical observations), auditory (for sound characteristics), and emotional (for interpreted feelings). Each array item should be a standalone, atomic description.
"""
system_prompt5 = """
Transform multimodal descriptions of a video character's emotional state (encompassing visual cues, auditory signals, and affective information) into a well-formatted JSON document. The output should categorize information into three clear sections: visual (appearance details), auditory (sound properties), and emotional (mood indicators). Each section must contain discrete, concise statements.
"""

example_prompt = """
EXAMPLE INPUT: 
In the video, we see a lady using a phone in an indoor environment. Her facial expression appears joyful, with the corners of her mouth turned up, indicating that she may be hearing some good news that pleases her. As time goes on, her eyes slightly squint, but her smile remains evident, accompanied by a slight nod, which is a natural reaction when people hear good news or something interesting. In the audio, the character's voice is loud, with a calm tone and a hint of laughter, indicating that the character is in a pleasant mood. In the text, the subtitle says, ""Then let's meet, okay? Well, goodbye."" This sentence may be an expression of the lady making an appointment to meet the other person over the phone. Based on the lady's joyful facial expression and smile in the video, as well as the character's pleasant tone and laughter in the audio, we can infer that this sentence is a positive response from the lady to the other person, indicating her willingness to meet. The lady's smile and nod further support this inference. Therefore, this sentence expresses the lady's happiness and anticipation.

EXAMPLE JSON OUTPUT:
{
  "visual": [
    "A woman is using a phone.",
    "The setting is indoors.",
    "Her facial expression is joyful.",
    "The corners of her mouth are upturned.",
    "Slight eye squinting over time.",
    "A persistent smile is evident.",
    "Accompanied by subtle nodding."
  ],
  "auditory": [
    "The voice is moderately loud.",
    "The tone remains calm.",
    "Laughter is perceptible.",
    "Subtitle text: 'Then let's meet, okay? Well, goodbye.'"
  ],
  "emotional": [
    "Likely receiving positive news.",
    "Microexpressions convey happiness.",
    "Nodding suggests agreement or interest.",
    "Vocal tone reflects a pleasant mood.",
    "Demonstrates willingness to meet.",
    "Facial expressions support positive affirmation.",
    "Interaction implies anticipation."
  ]
}
"""

system_prompt = system_prompt1+example_prompt


def save_dict_to_json(data, filename, indent=4, ensure_ascii=False):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        return True
    except Exception as e:
        print(f"保存JSON文件失败: {e}")
        return False

def load_json_to_dict(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None

def extract_vae_clue_from_text(user_prompt):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={
            'type': 'json_object'
        }
    )

    return json.loads(response.choices[0].message.content)

def extract_vae_clue_from_dict(user_dict, max_workers=50):
    result = {}
    lock = threading.Lock()

    def process(key, value):
        try:
            res = extract_vae_clue_from_text(value)
            with lock:
                result[key] = res
        except Exception as e:
            print(f"Error processing {key}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process, key, value) for key, value in user_dict.items()]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            pass

    return result

if __name__ == "__main__":
    INPUT_PATH = "/your/path/to/final-EMER-reason.csv"
    emer_dict = pd.read_csv(INPUT_PATH).set_index('name')['english'].to_dict()
    result = extract_vae_clue_from_dict(emer_dict)

    OUTPUT_PATH = "/your/path/to/EMER.json"
    save_dict_to_json(result, OUTPUT_PATH)
