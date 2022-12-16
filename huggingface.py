from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from transformers import pipeline

processor = WhisperProcessor.from_pretrained("openai/whisper-small")

device = torch.device("mps")

speech_recognizer = pipeline("automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30)
speech_recognizer.model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language = "zh", task = "transcribe")
result = speech_recognizer("/Users/username/Desktop/output.aac")["text"]
print(result)


