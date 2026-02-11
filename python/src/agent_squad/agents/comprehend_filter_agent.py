from typing import Optional, Callable, Any
from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.utils.logger import Logger
from .agent import Agent, AgentOptions
import boto3
from botocore.config import Config
import os
from dataclasses import dataclass


# Type alias for CheckFunction
CheckFunction = Callable[[str], str]

@dataclass
class ComprehendFilterAgentOptions(AgentOptions):
    enable_sentiment_check: bool = True
    enable_pii_check: bool = True
    enable_toxicity_check: bool = True
    sentiment_threshold: float = 0.7
    toxicity_threshold: float = 0.7
    allow_pii: bool = False
    language_code: str = 'en'
    region: Optional[str] = None
    client: Optional[Any] = None

class ComprehendFilterAgent(Agent):
    def __init__(self, options: ComprehendFilterAgentOptions):
        super().__init__(options)

        if options.client:
            self.comprehend_client = options.client
        else:
            if options.region:
                self.comprehend_client = boto3.client(
                    'comprehend',
                    region_name=options.region or os.environ.get('AWS_REGION')
                )
            else:
                self.comprehend_client = boto3.client('comprehend')

        self.custom_checks: list[CheckFunction] = []

        self.enable_sentiment_check = options.enable_sentiment_check
        self.enable_pii_check = options.enable_pii_check
        self.enable_toxicity_check = options.enable_toxicity_check
        self.sentiment_threshold = options.sentiment_threshold
        self.toxicity_threshold = options.toxicity_threshold
        self.allow_pii = options.allow_pii
        self.language_code = self.validate_language_code(options.language_code) or 'en'

        # Ensure at least one check is enabled
        if not any([self.enable_sentiment_check, self.enable_pii_check, self.enable_toxicity_check]):
            self.enable_toxicity_check = True

    async def process_request(self,
                              input_text: str,
                              user_id: str,
                              session_id: str,
                              chat_history: list[ConversationMessage],
                              additional_params: Optional[dict[str, str]] = None) -> Optional[ConversationMessage]:
        try:
            issues: list[str] = []

            # Run all checks
            sentiment_result = self.detect_sentiment(input_text) if self.enable_sentiment_check else None
            pii_result = self.detect_pii_entities(input_text) if self.enable_pii_check else None
            toxicity_result = self.detect_toxic_content(input_text) if self.enable_toxicity_check else None

            # Process results
            if self.enable_sentiment_check and sentiment_result:
                sentiment_issue = self.check_sentiment(sentiment_result)
                if sentiment_issue:
                    issues.append(sentiment_issue)

            if self.enable_pii_check and pii_result:
                pii_issue = self.check_pii(pii_result)
                if pii_issue:
                    issues.append(pii_issue)

            if self.enable_toxicity_check and toxicity_result:
                toxicity_issue = self.check_toxicity(toxicity_result)
                if toxicity_issue:
                    issues.append(toxicity_issue)

            # Run custom checks
            for check in self.custom_checks:
                custom_issue = await check(input_text)
                if custom_issue:
                    issues.append(custom_issue)

            if issues:
                Logger.warn(f"Content filter issues detected: {'; '.join(issues)}")
                return None  # Return None to indicate content should not be processed further

            # If no issues, return the original input as a ConversationMessage
            return ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": input_text}]
            )

        except Exception as error:
            Logger.error(f"Error in ComprehendContentFilterAgent:{str(error)}")
            raise error

    def add_custom_check(self, check: CheckFunction):
        self.custom_checks.append(check)

    def check_sentiment(self, result: dict[str, Any]) -> Optional[str]:
        if result['Sentiment'] == 'NEGATIVE' and result['SentimentScore']['Negative'] > self.sentiment_threshold:
            return f"Negative sentiment detected ({result['SentimentScore']['Negative']:.2f})"
        return None

    def check_pii(self, result: dict[str, Any]) -> Optional[str]:
        if not self.allow_pii and result.get('Entities'):
            return f"PII detected: {', '.join(e['Type'] for e in result['Entities'])}"
        return None

    def check_toxicity(self, result: dict[str, Any]) -> Optional[str]:
        toxic_labels = self.get_toxic_labels(result)
        if toxic_labels:
            return f"Toxic content detected: {', '.join(toxic_labels)}"
        return None

    def detect_sentiment(self, text: str) -> dict[str, Any]:
        return self.comprehend_client.detect_sentiment(
            Text=text,
            LanguageCode=self.language_code
        )

    def detect_pii_entities(self, text: str) -> dict[str, Any]:
        return self.comprehend_client.detect_pii_entities(
            Text=text,
            LanguageCode=self.language_code
        )

    def detect_toxic_content(self, text: str) -> dict[str, Any]:
        return self.comprehend_client.detect_toxic_content(
            TextSegments=[{"Text": text}],
            LanguageCode=self.language_code
        )

    def get_toxic_labels(self, toxicity_result: dict[str, Any]) -> list[str]:
        toxic_labels = []
        for result in toxicity_result.get('ResultList', []):
            for label in result.get('Labels', []):
                if label['Score'] > self.toxicity_threshold:
                    toxic_labels.append(label['Name'])
        return toxic_labels

    def set_language_code(self, language_code: str):
        validated_language_code = self.validate_language_code(language_code)
        if validated_language_code:
            self.language_code = validated_language_code
        else:
            raise ValueError(f"Invalid language code: {language_code}")

    @staticmethod
    def validate_language_code(language_code: Optional[str]) -> Optional[str]:
        if not language_code:
            return None

        valid_language_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ar', 'hi', 'ja', 'ko', 'zh', 'zh-TW']
        return language_code if language_code in valid_language_codes else None
