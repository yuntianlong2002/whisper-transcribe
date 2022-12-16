#!/Users/USERNAME/opt/anaconda3/envs/env_name/bin/python

# idea from https://piszek.com/2022/10/23/voice-memos-whisper/

import sqlite3
import os
import whisper
import argparse
import numpy as np
from whisper.utils import optional_int, optional_float, str2bool
import re
import json
import requests
import time

headers = {"Authorization": f"Bearer TOKEN"}
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    if response.status_code == 503:
        time.sleep(40)
        response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

# device = torch.device("mps")
model = whisper.load_model("small")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# The following are copied from whisper repo for tweaking
parser.add_argument("--language", type=str, default="zh", help="language spoken in the audio, specify None to perform language detection")
parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
parser.add_argument("--fp16", type=str2bool, default=False, help="whether to perform inference in fp16; True by default")

parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
# parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")

args = parser.parse_args().__dict__

temperature = args.pop("temperature")
temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
if temperature_increment_on_fallback is not None:
    temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
else:
    temperature = [temperature]

conn = sqlite3.connect('/Users/USERNAME/Library/Application Support/com.apple.voicememos/Recordings/CloudRecordings.db')
c = conn.cursor()
c.execute("select ZPATH, ZCUSTOMLABEL from ZCLOUDRECORDING")
res = c.fetchall()

for (filePath, fileComment) in res:
    print(fileComment)
    filename = fileComment+' ' +re.split('\.| ', filePath)[-2]+'.md'
    memopath = '/Users/USERNAME/Documents/username/voicememo/{}'.format(filename)

    if(os.path.getsize(filePath) < 10000000 and not os.path.isfile(memopath)):
        audio = whisper.load_audio(filePath)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        print(f"Detected language: {detected_language}")
        args["language"] = detected_language
        if detected_language == "en":
            # TODO: convert to AAC format first
            data = query(filePath)
            print(data)
        else:
            result = model.transcribe(filePath, temperature=temperature, **args)
            print(result["text"])
            with open(memopath, 'w') as f:
                f.write(result["text"])
conn.close()
