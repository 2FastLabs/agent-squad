#!/usr/bin/env python3
"""
Streamlit Chat Demo for Multi-Agent Orchestrator

A simple chat interface demonstrating multi-agent conversation.

Setup:
    pip install multi-agent-orchestrator streamlit openai
    export OPENAI_API_KEY="sk-..."

Run:
    streamlit run chat_demo.py
"""

import streamlit as st
from agent_squad import MultiAgentOrchestrator, Agent


# Configure agents
coding_agent = Agent(
    name="CodingAssistant",
    description="Expert software developer helping with coding tasks",
    instructions="You are a helpful coding assistant. Provide clear, concise code examples.",
)

research_agent = Agent(
    name="ResearchAssistant",
    description="Expert researcher helping with information synthesis",
    instructions="You are a helpful research assistant. Summarize information clearly.",
)


def main():
    st.set_page_config(page_title="Multi-Agent Chat", page_icon="🤖")
    st.title("🤖 Multi-Agent Orchestrator Chat")

    st.markdown("""
    This demo shows the **Multi-Agent Orchestrator** in action.
    Select an agent and start chatting!
    """)

    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(agents=[coding_agent, research_agent])

    # Agent selection
    agent_names = [agent.name for agent in orchestrator.agents]
    selected_agent = st.selectbox("Choose an agent:", agent_names)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask something..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner(f"{selected_agent} is thinking..."):
                response = orchestrator.process_request(
                    prompt, agent_name=selected_agent
                )
                st.markdown(response)

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
