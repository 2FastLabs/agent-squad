"""
Code for Classifier.
"""
from .classifier import Classifier, ClassifierResult, ClassifierCallbacks

try:
    from .bedrock_classifier import BedrockClassifier, BedrockClassifierOptions
    _AWS_AVAILABLE = True
except Exception as e:
    _AWS_AVAILABLE = False

try:
    from .anthropic_classifier import AnthropicClassifier, AnthropicClassifierOptions
    _ANTHROPIC_AVAILABLE = True
except Exception as e:
    _ANTHROPIC_AVAILABLE = False

try:
    from .openai_classifier import OpenAIClassifier, OpenAIClassifierOptions
    _OPENAI_AVAILABLE = True
except Exception as e:
    _OPENAI_AVAILABLE = False

try:
    from .minimax_classifier import MiniMaxClassifier, MiniMaxClassifierOptions
    _MINIMAX_AVAILABLE = True
except Exception as e:
    _MINIMAX_AVAILABLE = False

__all__ = [
    "Classifier",
    "ClassifierResult",
    'ClassifierCallbacks'
]

if _AWS_AVAILABLE:
    __all__.extend([
        "BedrockClassifier",
        "BedrockClassifierOptions"
    ])

if _ANTHROPIC_AVAILABLE:
    __all__.extend([
        "AnthropicClassifier",
        "AnthropicClassifierOptions"
    ])

if _OPENAI_AVAILABLE:
    __all__.extend([
        "OpenAIClassifier",
        "OpenAIClassifierOptions"
    ])

if _MINIMAX_AVAILABLE:
    __all__.extend([
        "MiniMaxClassifier",
        "MiniMaxClassifierOptions"
    ])