---
title: Storage overview
description: An overview of conversation storage options in the Agent Squad System
---

The Agent Squad System offers flexible storage options for maintaining conversation history. This allows the system to preserve context across multiple interactions and enables agents to provide more coherent and contextually relevant responses.

## Key Concepts

- Each conversation is uniquely identified by a combination of `userId`, `sessionId`, and `agentId`.
- The storage system saves both user messages and assistant responses.
- Different storage backends are supported through the `ConversationStorage` interface.

## Available Storage Options

1. **In-Memory Storage**:
   - Ideal for development, testing, or scenarios where persistence isn't required.
   - Quick and efficient for short-lived sessions.

2. **DynamoDB Storage**:
   - Provides persistent storage for production environments.
   - Allows for scalable and durable conversation history storage.

3. **SQL Storage**:
    - Offers persistent storage using SQLite or Turso databases.
    - When you need local-first development with remote deployment options

4. **Summarizing Storage**:
   - Wraps any storage backend and automatically compresses long histories via an LLM summarizer.
   - Keeps agent context small without losing the thread of the conversation.

5. **Custom Storage Solutions**:
   - The system allows for implementation of custom storage options to meet specific needs.

## Choosing the Right Storage Option

- Use In-Memory Storage for development, testing, or when persistence between application restarts is not necessary.
- Choose DynamoDB Storage for production environments where conversation history needs to be preserved long-term or across multiple instances of your application.
- Consider SQL Storage for a balance between simplicity and scalability, supporting both local and remote databases.
- Use Summarizing Storage to keep agent context small in long-running conversations by wrapping any backend with an LLM summarizer.
- Implement a custom storage solution if you have specific requirements not met by the provided options.

## Next Steps

- Learn more about [In-Memory Storage](/agent-squad/storage/in-memory)
- Explore [DynamoDB Storage](/agent-squad/storage/dynamodb) for persistent storage
- Explore [SQL Storage](/agent-squad/storage/sql) for persistent storage using SQLite or Turso.
- Use [Summarizing Storage](/agent-squad/storage/summarizing) to compress long histories automatically.
- Discover how to [implement custom storage solutions](/agent-squad/storage/custom)

By leveraging these storage options, you can ensure that your Agent Squad System maintains the necessary context for coherent and effective conversations across various use cases and deployment scenarios.