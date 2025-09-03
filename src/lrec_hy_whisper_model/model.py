import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizer,WhisperFeatureExtractor, AutoProcessor, pipeline

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
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu" # TODO: Warn user if we're on the CPU
        self.model.to(self.device)
        return pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.tokenizer,
            feature_extractor = self.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            return_timestamps=self.return_timestamps,
            device = self.device     
        )
    
    def transcribe_longform_audio_with_pipeline(self, audio_source): 
        return self.pipeline(audio_source.audio_array)

    def transcribe_longform_audio_with_generate(self, audio_source): 
        inputs = self.processor(
            audio_source.audio_array, 
            sampling_rate=audio_source.sampling_rate, 
            return_tensors="pt", 
            truncation=False, 
            padding="longest", 
            return_attention_mask=True,
        )
        inputs = inputs.to(self.device)

        generate_config = {
            "return_timestamps": True, # Required for long-form transcription
        }
        
        pred_ids = self.model.generate(**inputs, **generate_config)
        return self.processor.batch_decode(pred_ids, skip_special_tokens=False, decode_with_timestamps=True)


class LrecHyAsr(TransformersWhisperAsr): 
    model_id = "./models/whisper_v3_lrec"

# TODO: Add others (mini, etc.)

