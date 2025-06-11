from transformers import AutoTokenizer, M2M100ForConditionalGeneration
import torch
from langdetect import detect, LangDetectException
import logging

logger = logging.getLogger(__name__)

# Device configuration for Mac
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Translation using device: {device}")

# Initialize translation models
try:
    tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)
    logger.info("Translation model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load translation model: {str(e)}")
    raise

def translate_text(text: str, target_lang: str = "en") -> str:
    try:
        # Detect source language
        src_lang = detect(text)
        
        # Return original if already in target language
        if src_lang == target_lang:
            return text
            
        # Set source language
        tokenizer.src_lang = src_lang
        
        # Tokenize and translate
        encoded = tokenizer(text, return_tensors="pt").to(device)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(target_lang)
        )
        
        # Decode and return translation
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        logger.debug(f"Translated from {src_lang} to {target_lang}")
        
        return translation
        
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {str(e)}")
        return text
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return text