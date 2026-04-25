## FAQ

### General

**What is Agent Squad?**
Agent Squad is a flexible, lightweight open-source framework for orchestrating multiple AI agents to handle complex conversations. It intelligently routes queries and maintains context across interactions.

**What languages are supported?**
Agent Squad is fully implemented in both **Python** and **TypeScript**, with feature parity between the two implementations.

**Where was this project previously hosted?**
Agent Squad was previously hosted at `awslabs/agent-squad` and has moved to `2fastlabs/agent-squad`. Please update your bookmarks and dependencies accordingly.

### Getting Started

**How do I install Agent Squad?**
```sh
# Python
pip install agent-squad

# TypeScript
npm install agent-squad
```

**What's the basic workflow?**
1. Define your agents with specific capabilities
2. Set up a classifier to route queries
3. Create an orchestrator to manage conversations
4. The system automatically routes queries to the best agent

### Agents

**What agents are available out of the box?**
Agent Squad provides pre-built agents and classifiers for quick deployment. You can also create custom agents by implementing the agent interface.

**What is SupervisorAgent?**
SupervisorAgent enables sophisticated team coordination between multiple specialized agents. It implements an "agent-as-tools" architecture, allowing a lead agent to coordinate a team of specialized agents in parallel.

**Can I create custom agents?**
Yes. Agent Squad's extensible architecture lets you integrate new agents or customize existing ones to fit your specific needs.

### Conversation Management

**How does context management work?**
Agent Squad maintains and utilizes conversation context across multiple agents for coherent interactions. The system stores conversation history and uses it for intelligent routing.

**Can I customize conversation storage?**
Yes. Agent Squad allows easy integration of custom conversation message storage solutions.

### Deployment

**Where can I deploy Agent Squad?**
Agent Squad supports universal deployment - from AWS Lambda to your local environment or any cloud platform.

**Does it support streaming responses?**
Yes. Agent Squad supports both streaming and non-streaming responses from different agents.

### Troubleshooting

**Import errors after installation**
- Ensure you're using the correct package name: `agent-squad`
- For Python, verify your virtual environment is activated
- For TypeScript, check that `@2fastlabs/agent-squad` is in your `package.json`

**Classifier not routing correctly**
- Verify your agents' characteristics are properly defined
- Check that the classifier has sufficient training data
- Review the intent classification logic for your use case

**Context not persisting between agents**
- Ensure you're using the same orchestrator instance
- Verify your conversation storage is properly configured
- Check that messages are being saved after each interaction

**Performance issues with multiple agents**
- Consider using streaming responses for better UX
- Review agent initialization - lazy load agents if not all are needed
- Monitor LLM API rate limits and adjust concurrency accordingly

