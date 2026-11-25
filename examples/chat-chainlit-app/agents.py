from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions, AgentCallbacks
from ollamaAgent import OllamaAgent, OllamaAgentOptions
import asyncio

import chainlit as cl

class ChainlitAgentCallbacks(AgentCallbacks):
    def on_llm_new_token(self, token: str) -> None:
        asyncio.run(cl.user_session.get("current_msg").stream_token(token))

def create_tech_agent():
    return BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Tech Agent",
        streaming=True,
        description="Specializes in technology areas including software development, hardware, AI, cybersecurity, blockchain, cloud computing, emerging tech innovations, and pricing/costs related to technology products and services.",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        callbacks=ChainlitAgentCallbacks()
    ))

def create_travel_agent():
    return BedrockLLMAgent(BedrockLLMAgentOptions(
        name="Travel Agent",
        streaming=True,
        description="Experienced Travel Agent sought to create unforgettable journeys for clients. Responsibilities include crafting personalized itineraries, booking flights, accommodations, and activities, and providing expert travel advice. Must have excellent communication skills, destination knowledge, and ability to manage multiple bookings. Proficiency in travel booking systems and a passion for customer service required",
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        callbacks=ChainlitAgentCallbacks()
    ))

def create_health_agent():
    return OllamaAgent(OllamaAgentOptions(
       name="Health Agent",
        model_id="llama3.1:latest",
        description="Specializes in health and wellness, including nutrition, fitness, mental health, and disease prevention. Provides personalized health advice, creates wellness plans, and offers resources for self-care. Must have a strong understanding of human anatomy, physiology, and medical terminology. Proficiency in health coaching techniques and a commitment to promoting overall well-being required.",
        streaming=True,
        callbacks=ChainlitAgentCallbacks()
    ))
