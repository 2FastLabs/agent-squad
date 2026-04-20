import unittest
from unittest.mock import Mock
from typing import Dict, Any

from agent_squad.types import ConversationMessage, ParticipantRole
from agent_squad.agents import ComprehendFilterAgent, ComprehendFilterAgentOptions

class TestComprehendFilterAgent(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create mock comprehend client
        self.mock_comprehend_client = Mock()

        # Setup default positive responses
        self.mock_comprehend_client.detect_sentiment.return_value = {
            'Sentiment': 'POSITIVE',
            'SentimentScore': {
                'Positive': 0.9,
                'Negative': 0.1,
                'Neutral': 0.0,
                'Mixed': 0.0
            }
        }

        self.mock_comprehend_client.detect_pii_entities.return_value = {
            'Entities': []
        }

        self.mock_comprehend_client.detect_toxic_content.return_value = {
            'ResultList': [{
                'Labels': []
            }]
        }

        # Create agent instance
        self.agent = ComprehendFilterAgent(
            ComprehendFilterAgentOptions(
                name="Test Filter Agent",
                description="Test agent for filtering content",
                client=self.mock_comprehend_client
            )
        )

    async def test_initialization(self):
        """Test agent initialization and configuration"""
        self.assertEqual(self.agent.name, "Test Filter Agent")
        self.assertEqual(self.agent.description, "Test agent for filtering content")
        self.assertTrue(self.agent.enable_sentiment_check)
        self.assertTrue(self.agent.enable_pii_check)
        self.assertTrue(self.agent.enable_toxicity_check)
        self.assertEqual(self.agent.language_code, "en")

    async def test_process_clean_content(self):
        """Test processing clean content passes through filters"""
        input_text = "Hello, this is a friendly message!"

        response = await self.agent.process_request(
            input_text=input_text,
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )

        self.assertIsNotNone(response)
        self.assertIsInstance(response, ConversationMessage)
        self.assertEqual(response.role, ParticipantRole.ASSISTANT.value)
        self.assertEqual(response.content[0]["text"], input_text)

    async def test_negative_sentiment_blocking(self):
        """Test that highly negative content is blocked"""
        # Configure mock for negative sentiment
        self.mock_comprehend_client.detect_sentiment.return_value = {
            'Sentiment': 'NEGATIVE',
            'SentimentScore': {
                'Positive': 0.0,
                'Negative': 0.9,
                'Neutral': 0.1,
                'Mixed': 0.0
            }
        }

        response = await self.agent.process_request(
            input_text="I hate everything!",
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )

        self.assertIsNone(response)
        self.mock_comprehend_client.detect_sentiment.assert_called_once()

    async def test_pii_detection_blocking(self):
        """Test that content with PII is blocked"""
        # Configure mock for PII detection
        self.mock_comprehend_client.detect_pii_entities.return_value = {
            'Entities': [
                {'Type': 'EMAIL', 'Score': 0.99},
                {'Type': 'PHONE', 'Score': 0.95}
            ]
        }

        response = await self.agent.process_request(
            input_text="Contact me at test@email.com",
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )

        self.assertIsNone(response)
        self.mock_comprehend_client.detect_pii_entities.assert_called_once()

    async def test_toxic_content_blocking(self):
        """Test that toxic content is blocked"""
        # Configure mock for toxic content
        self.mock_comprehend_client.detect_toxic_content.return_value = {
            'ResultList': [{
                'Labels': [
                    {'Name': 'HATE_SPEECH', 'Score': 0.95}
                ]
            }]
        }

        response = await self.agent.process_request(
            input_text="Some toxic content here",
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )

        self.assertIsNone(response)
        self.mock_comprehend_client.detect_toxic_content.assert_called_once()

    async def test_custom_check(self):
        """Test custom check functionality"""
        async def custom_check(text: str) -> str:
            if "banned" in text.lower():
                return "Contains banned word"
            return None

        self.agent.add_custom_check(custom_check)

        response = await self.agent.process_request(
            input_text="This contains a banned word",
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )

        self.assertIsNone(response)

    async def test_language_code_validation(self):
        """Test language code validation and setting"""
        # Test valid language code
        self.agent.set_language_code("es")
        self.assertEqual(self.agent.language_code, "es")

        # Test invalid language code
        with self.assertRaises(ValueError):
            self.agent.set_language_code("invalid")

    async def test_allow_pii_configuration(self):
        """Test PII allowance configuration"""
        # Create new agent instance with PII allowed
        agent_with_pii = ComprehendFilterAgent(
            ComprehendFilterAgentOptions(
                name="Test Filter Agent",
                description="Test agent for filtering content",
                client=self.mock_comprehend_client,
                allow_pii=True
            )
        )

        # Configure mock for PII detection
        self.mock_comprehend_client.detect_pii_entities.return_value = {
            'Entities': [
                {'Type': 'EMAIL', 'Score': 0.99}
            ]
        }

        response = await agent_with_pii.process_request(
            input_text="Contact me at test@email.com",
            user_id="test_user",
            session_id="test_session",
            chat_history=[]
        )

        self.assertIsNotNone(response)
        self.assertEqual(response.content[0]["text"], "Contact me at test@email.com")

    async def test_error_handling(self):
        """Test error handling in process_request"""
        # Configure mock to raise an exception
        self.mock_comprehend_client.detect_sentiment.side_effect = Exception("Test error")

        with self.assertRaises(Exception) as context:
            await self.agent.process_request(
                input_text="Hello",
                user_id="test_user",
                session_id="test_session",
                chat_history=[]
            )

        self.assertTrue("Test error" in str(context.exception))

    async def test_threshold_configuration(self):
        """Test custom threshold configurations"""
        agent = ComprehendFilterAgent(
            ComprehendFilterAgentOptions(
                name="Test Filter Agent",
                description="Test agent for filtering content",
                client=self.mock_comprehend_client,
                sentiment_threshold=0.5,
                toxicity_threshold=0.8
            )
        )

        self.assertEqual(agent.sentiment_threshold, 0.5)
        self.assertEqual(agent.toxicity_threshold, 0.8)

    async def test_initialization_without_client_uses_comprehend_client(self):
        """Test that comprehend_client is correctly set when no client is passed.

        This regression test verifies the fix for issue #359 where passing
        client=None would cause an AttributeError because the boto3 client
        was incorrectly assigned to self.client instead of self.comprehend_client.
        """
        from unittest.mock import patch, MagicMock

        mock_boto_client = MagicMock()
        with patch('boto3.client', return_value=mock_boto_client) as mock_boto:
            # Test with region specified
            agent_with_region = ComprehendFilterAgent(
                ComprehendFilterAgentOptions(
                    name="Test Filter Agent",
                    description="Test agent",
                    client=None,
                    region="us-east-1"
                )
            )
            # Verify comprehend_client attribute exists and is set correctly
            self.assertTrue(hasattr(agent_with_region, 'comprehend_client'))
            self.assertEqual(agent_with_region.comprehend_client, mock_boto_client)
            mock_boto.assert_called_with('comprehend', region_name='us-east-1')

        with patch('boto3.client', return_value=mock_boto_client) as mock_boto:
            # Test without region (uses default)
            agent_without_region = ComprehendFilterAgent(
                ComprehendFilterAgentOptions(
                    name="Test Filter Agent",
                    description="Test agent",
                    client=None,
                    region=None
                )
            )
            # Verify comprehend_client attribute exists and is set correctly
            self.assertTrue(hasattr(agent_without_region, 'comprehend_client'))
            self.assertEqual(agent_without_region.comprehend_client, mock_boto_client)
            mock_boto.assert_called_with('comprehend')

if __name__ == '__main__':
    unittest.main()