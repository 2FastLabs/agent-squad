---
title: Multi-lingual Routing Pattern
description: Multi-lingual Routing Pattern  using the Agent Squad framework
---


By integrating language-specific agents, the Agent Squad can provide multi-lingual support, enabling users to interact with the system in their preferred language while maintaining consistent experiences.

## Key Components

1. **Language Detection**
   - Classifier identifies query language
   - Routes to appropriate language-specific agent
   - Maintains context across languages

2. **Language-Specific Agents**
   - Dedicated agents for each supported language
   - Specialized in language-specific responses
   - Consistent response quality across languages

3. **Dynamic Language Routing**
   - Automatic routing based on detected language
   - Seamless language switching
   - Maintains conversation context

## Implementation Example

```typescript
// French language agent
orchestrator.addAgent(
  new BedrockLLMAgent({
    name: "Text Summarization Agent for French Language",
    modelId: "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    description: "This is a very simple text summarization agent for french language.",
    streaming: true,
    inferenceConfig: {
      temperature: 0.0,
    },
  })
);

// English language agent
orchestrator.addAgent(
  new BedrockLLMAgent({
    name: "Text Summarization Agent English Language",
    modelId: "mistral.mistral-small-2402-v1:0",
    description: "This is a very simple text summarization agent for english language.",
    streaming: true,
    inferenceConfig: {
      temperature: 0.0,
    }
  })
);
```

## Implementation Notes
- Models shown are for illustration
- Any suitable LLM can be substituted
- Principle remains consistent across different models
- Configure based on language-specific requirements

## Benefits
- Native language support
- Consistent user experience
- Scalable language coverage
- Maintainable language-specific logic