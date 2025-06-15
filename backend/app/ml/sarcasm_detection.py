from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import logging
import re
from typing import Dict, List, Optional
import numpy as np

# Pre-trained model
MODEL_PATH = "app/ml/sarcasm_sentiment_model"

# Configure logging
logger = logging.getLogger(__name__)

# Device configuration
if torch.backends.mps.is_available():
    device = 0  # MPS on Mac
    logger.info("Using device: MPS")
elif torch.cuda.is_available():
    device = 0  # CUDA
    logger.info("Using device: CUDA")
else:
    device = -1  # CPU
    logger.info("Using device: CPU")

class AdvancedSarcasmDetector:
    def __init__(self):
        logger.info("Initializing Advanced Sarcasm Detector...")
        
        # Load multiple models for ensemble approach
        self.models = self._load_models()
        
        # Sarcasm indicators and patterns
        self.sarcasm_indicators = {
            'exaggeration_words': [
                'absolutely', 'totally', 'completely', 'perfectly', 'utterly',
                'extremely', 'incredibly', 'amazingly', 'super', 'mega'
            ],
            'contradiction_phrases': [
                'oh great', 'just great', 'how wonderful', 'fantastic',
                'brilliant', 'perfect', 'lovely', 'charming'
            ],
            'sarcastic_punctuation': [
                '!!!', '...', '!!', '???', '!?', '?!'
            ],
            'intensifiers': [
                'so', 'very', 'really', 'quite', 'rather', 'pretty'
            ]
        }
        
        logger.info("Advanced Sarcasm Detector initialized!")

    def _load_models(self) -> Dict:
        """Load multiple models for ensemble prediction"""
        models = {}
        
        try:
            # Model 1: Twitter RoBERTa
            logger.info("Loading Twitter RoBERTa model...")
            # tokenizer1 = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-irony")
            # model1 = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-irony")
            
            # Some changes here. 
            tokenizer1 = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                use_auth_token=False
            )

            model1 = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                local_files_only=True,
                use_auth_token=False
)
            
            models['twitter_roberta'] = pipeline(
                "text-classification", 
                model=model1, 
                tokenizer=tokenizer1, 
                device=device,
                truncation=True,
                max_length=128
            )
            
            # Model 2: Alternative sarcasm detection model
            logger.info("Loading alternative sarcasm model...")
            try:
                tokenizer2 = AutoTokenizer.from_pretrained("helinivan/english-sarcasm-detector")
                model2 = AutoModelForSequenceClassification.from_pretrained("helinivan/english-sarcasm-detector")
                models['sarcasm_detector'] = pipeline(
                    "text-classification",
                    model=model2,
                    tokenizer=tokenizer2,
                    device=device,
                    truncation=True,
                    max_length=128
                )
            except Exception as e:
                logger.warning(f"Could not load second model: {e}")
            
            # Model 3: Sentiment-based approach
            logger.info("Loading sentiment model for context...")
            try:
                models['sentiment'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=device,
                    truncation=True,
                    max_length=128
                )
            except Exception as e:
                logger.warning(f"Could not load sentiment model: {e}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
            
        return models

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Remove URLs and mentions
        text = re.sub(r"http\S+|@\w+", "", text)
        
        # Normalize repeated punctuation
        text = re.sub(r'[!]{2,}', '!!', text)
        text = re.sub(r'[?]{2,}', '??', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text

    def _extract_linguistic_features(self, text: str) -> Dict:
        """Extract linguistic features that indicate sarcasm"""
        text_lower = text.lower()
        features = {
            'has_exaggeration': any(word in text_lower for word in self.sarcasm_indicators['exaggeration_words']),
            'has_contradiction_phrase': any(phrase in text_lower for phrase in self.sarcasm_indicators['contradiction_phrases']),
            'has_sarcastic_punctuation': any(punct in text for punct in self.sarcasm_indicators['sarcastic_punctuation']),
            'has_intensifiers': any(word in text_lower for word in self.sarcasm_indicators['intensifiers']),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'ellipsis_count': text.count('...'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'word_count': len(text.split())
        }
        
        # Calculate feature score
        feature_score = 0
        if features['has_exaggeration']: feature_score += 0.3
        if features['has_contradiction_phrase']: feature_score += 0.4
        if features['has_sarcastic_punctuation']: feature_score += 0.2
        if features['has_intensifiers']: feature_score += 0.1
        if features['exclamation_count'] > 1: feature_score += 0.2
        if features['caps_ratio'] > 0.3: feature_score += 0.2
        
        features['linguistic_score'] = min(feature_score, 1.0)
        
        return features

    def _context_analysis(self, text: str) -> Dict:
        """Analyze context for sarcasm detection"""
        context = {
            'is_positive_words_negative_context': False,
            'has_contrast': False,
            'sentiment_mismatch': False
        }
        
        # Check for positive words in potentially negative contexts
        positive_words = ['great', 'wonderful', 'perfect', 'amazing', 'fantastic', 'brilliant']
        negative_contexts = ['fail', 'broke', 'wrong', 'bad', 'terrible', 'awful', 'stupid']
        
        text_lower = text.lower()
        has_positive = any(word in text_lower for word in positive_words)
        has_negative = any(word in text_lower for word in negative_contexts)
        
        if has_positive and has_negative:
            context['is_positive_words_negative_context'] = True
            context['has_contrast'] = True
        
        # Check for sentiment analysis if available
        if 'sentiment' in self.models:
            try:
                sentiment_result = self.models['sentiment'](text[:512])[0]  # Truncate for safety
                # If positive words but negative sentiment, could be sarcasm
                if has_positive and sentiment_result['label'] == 'LABEL_0':  # Negative
                    context['sentiment_mismatch'] = True
            except Exception as e:
                logger.warning(f"Sentiment analysis error: {e}")
        
        return context

    def detect_sarcasm(self, text: str) -> Dict:
        """Advanced sarcasm detection with ensemble approach"""
        if not text or len(text.strip()) < 3:
            return {"is_sarcastic": False, "confidence": 0.0, "details": "Text too short"}
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Extract features
        linguistic_features = self._extract_linguistic_features(text)
        context_analysis = self._context_analysis(text)
        
        # Get predictions from all available models
        model_predictions = []
        model_details = {}
        
        # Twitter RoBERTa prediction
        if 'twitter_roberta' in self.models:
            try:
                result = self.models['twitter_roberta'](processed_text[:512])[0]  # Truncate
                # Convert label to boolean (IRONY = sarcastic)
                is_sarcastic = result['label'] == 'IRONY'
                confidence = result['score'] if is_sarcastic else 1 - result['score']
                model_predictions.append(confidence if is_sarcastic else 0)
                model_details['twitter_roberta'] = {
                    'label': result['label'],
                    'score': result['score'],
                    'is_sarcastic': is_sarcastic
                }
            except Exception as e:
                logger.warning(f"Twitter RoBERTa error: {e}")
        
        # Alternative sarcasm detector
        if 'sarcasm_detector' in self.models:
            try:
                result = self.models['sarcasm_detector'](processed_text[:512])[0]  # Truncate
                # This model might have different labels, adjust accordingly
                is_sarcastic = result['label'] in ['SARCASM', '1', 'sarcastic']
                confidence = result['score'] if is_sarcastic else 1 - result['score']
                model_predictions.append(confidence if is_sarcastic else 0)
                model_details['sarcasm_detector'] = {
                    'label': result['label'],
                    'score': result['score'],
                    'is_sarcastic': is_sarcastic
                }
            except Exception as e:
                logger.warning(f"Sarcasm detector error: {e}")
        
        # Combine predictions using weighted ensemble
        if model_predictions:
            model_confidence = np.mean(model_predictions) if model_predictions else 0.0
        else:
            model_confidence = 0.0
        
        # Combine with linguistic features
        feature_weight = 0.3
        model_weight = 0.7
        
        final_confidence = (
            model_weight * model_confidence + 
            feature_weight * linguistic_features['linguistic_score']
        )
        
        # Context adjustments
        if context_analysis['sentiment_mismatch']:
            final_confidence = min(final_confidence + 0.15, 1.0)
        if context_analysis['has_contrast']:
            final_confidence = min(final_confidence + 0.15, 1.0)
        
        # Determine if sarcastic (threshold can be adjusted)
        threshold = 0.6  # Higher threshold to reduce false positives
        is_sarcastic = final_confidence > threshold
        
        return {
            "is_sarcastic": is_sarcastic,
            "confidence": round(final_confidence, 3),
            "details": {
                "model_predictions": model_details,
                "linguistic_features": linguistic_features,
                "context_analysis": context_analysis,
                "model_confidence": round(model_confidence, 3),
                "linguistic_score": round(linguistic_features['linguistic_score'], 3)
            }
        }