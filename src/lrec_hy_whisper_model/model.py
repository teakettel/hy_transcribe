import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer,WhisperFeatureExtractor, pipeline

# TODO: Typing
# TODO: Pick a good batch size :)

DEFAULT_WHISPER_MODEL_ID = "openai/whisper-base"
ARMENIAN_LANGUAGE_ID = 'hy'

class TransformersWhisperAsr:
    model_id = DEFAULT_WHISPER_MODEL_ID
    language_id = ARMENIAN_LANGUAGE_ID
    chunk_length_s = 30
    batch_size = 16 
    return_timestamps = True 
    
    def __init__(self):
        self.pipeline = self._load(self.model_id, self.language_id)

    def _load(self, model_id, language_id): 
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_id, language=language_id) 
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        device = "cuda:0" if torch.cuda.is_available() else "cpu" # TODO: check if I want this this way
        return pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor = self.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            return_timestamps=self.return_timestamps,
            device = device     
        )
    
    def transcribe_longform_audio(self, audio_source): 
        return self.pipeline(audio_source.audio_array)

class LrecHyAsr(TransformersWhisperAsr): 
    model_id = "./models/whisper_v3_lrec"
 
# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
# from datasets import Audio, load_dataset


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"

# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
# )
# model.to(device)

# processor = AutoProcessor.from_pretrained(model_id)

# dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
# sample = dataset[0]["audio"]

# inputs = processor(
#     sample["array"],
#     sampling_rate=sample["sampling_rate"],
#     return_tensors="pt",
#     truncation=False,
#     padding="longest",
#     return_attention_mask=True,
# )
# inputs = inputs.to(device, dtype=torch_dtype)

# gen_kwargs = {
#     "max_new_tokens": 448,
#     "num_beams": 1,
#     "condition_on_prev_tokens": False,
#     "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
#     "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
#     "logprob_threshold": -1.0,
#     "no_speech_threshold": 0.6,
#     "return_timestamps": True,
# }

# pred_ids = model.generate(**inputs, **gen_kwargs)
# pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True, decode_with_timestamps=False)

# print(pred_text)

