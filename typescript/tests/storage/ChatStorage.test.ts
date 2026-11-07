import { InMemoryChatStorage } from "../../src/storage/memoryChatStorage";
import { ConversationMessage, ParticipantRole } from "../../src/types";
describe("InMemoryChatStorage", () => {
  let storage: InMemoryChatStorage;

  function delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  beforeEach(() => {
    storage = new InMemoryChatStorage();
  });

  const createMessage = (role: ParticipantRole, text: string): ConversationMessage => ({
    role,
    content: [{ text }],
  });

   test("saveChatMessage should maintain alternating roles", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";

    const message1 = createMessage(ParticipantRole.USER, "Hello");
    const message2 = createMessage(ParticipantRole.ASSISTANT, "Hi there");
    const message3 = createMessage(ParticipantRole.USER, "How are you?");
    const message4 = createMessage(
      ParticipantRole.ASSISTANT,
      "I'm doing well, thanks!"
    );

    await storage.saveChatMessage(userId, sessionId, agentId, message1);
    await storage.saveChatMessage(userId, sessionId, agentId, message2);
    await storage.saveChatMessage(userId, sessionId, agentId, message3);
    await storage.saveChatMessage(userId, sessionId, agentId, message4);

    const conversation = await storage.fetchChat(userId, sessionId, agentId);

    expect(conversation).toHaveLength(4);
    expect(conversation[0].role).toBe(ParticipantRole.USER);
    expect(conversation[1].role).toBe(ParticipantRole.ASSISTANT);
    expect(conversation[2].role).toBe(ParticipantRole.USER);
    expect(conversation[3].role).toBe(ParticipantRole.ASSISTANT);
  });

  test("saveChatMessage should maintain message order", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";

    const messages = [
      createMessage(ParticipantRole.USER, "Message 1"),
      createMessage(ParticipantRole.ASSISTANT, "Message 2"),
      createMessage(ParticipantRole.USER, "Message 3"),
      createMessage(ParticipantRole.ASSISTANT, "Message 4"),
    ];

    for (const message of messages) {
      await storage.saveChatMessage(userId, sessionId, agentId, message);
    }

    const conversation = await storage.fetchChat(userId, sessionId, agentId);

    expect(conversation).toHaveLength(4);
    expect(conversation.map((m) => m.content?.[0]?.text)).toEqual([
      "Message 1",
      "Message 2",
      "Message 3",
      "Message 4",
    ]);
  });

  test("saveChatMessage should not allow consecutive messages with the same role", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";

    const message1 = createMessage(ParticipantRole.USER, "Hello");
    const message2 = createMessage(ParticipantRole.USER, "Are you there?");

    await storage.saveChatMessage(userId, sessionId, agentId, message1);
    await storage.saveChatMessage(userId, sessionId, agentId, message2);

    const conversation = await storage.fetchChat(userId, sessionId, agentId);

    expect(conversation).toHaveLength(1);
    expect(conversation[0].content?.[0]?.text).toBe("Hello");
  });

  test("saveChatMessage should allow alternating roles", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";

    const message1 = createMessage(ParticipantRole.USER, "Hello");
    const message2 = createMessage(ParticipantRole.ASSISTANT, "Hi there");
    const message3 = createMessage(ParticipantRole.USER, "How are you?");

    await storage.saveChatMessage(userId, sessionId, agentId, message1);
    await storage.saveChatMessage(userId, sessionId, agentId, message2);
    await storage.saveChatMessage(userId, sessionId, agentId, message3);

    const conversation = await storage.fetchChat(userId, sessionId, agentId);

    expect(conversation).toHaveLength(3);
    expect(conversation[0].content?.[0]?.text).toBe("Hello");
    expect(conversation[1].content?.[0]?.text).toBe("Hi there");
    expect(conversation[2].content?.[0]?.text).toBe("How are you?");
  });

  test("fetchAllChats should return messages from all agents for a session", async () => {

    const userId = "user1";
    const sessionId = "session1";
    const agent1 = "agent1";
    const agent2 = "agent2";

    await storage.saveChatMessage(
      userId,
      sessionId,
      agent1,
      createMessage(ParticipantRole.USER, "Hello Agent 1")
    );
    await storage.saveChatMessage(
      userId,
      sessionId,
      agent1,
      createMessage(ParticipantRole.ASSISTANT, "Hi from Agent 1")
    );
    await storage.saveChatMessage(
      userId,
      sessionId,
      agent2,
      createMessage(ParticipantRole.USER, "Hello Agent 2")
    );
    await storage.saveChatMessage(
      userId,
      sessionId,
      agent2,
      createMessage(ParticipantRole.ASSISTANT, "Hi from Agent 2")
    );

    const allChats = await storage.fetchAllChats(userId, sessionId);

    expect(allChats).toHaveLength(4);

    expect(allChats.some(chat =>
      chat.content?.some(item => 
        item.text === "Hello Agent 1" || 
        item.text === "[agent1] Hi from Agent 1" || 
        item.text === "Hello Agent 2" || 
        item.text === "[agent2] Hi from Agent 2"
      )
    )).toBe(true);
  });

  test("fetchAllChats should return all messages when no maxHistorySize is set", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";

    for (let i = 0; i < 5; i++) {
      await storage.saveChatMessage(
        userId,
        sessionId,
        agentId,
        createMessage(ParticipantRole.USER, `User message ${i + 1}`)
      );
      await storage.saveChatMessage(
        userId,
        sessionId,
        agentId,
        createMessage(ParticipantRole.ASSISTANT, `Assistant message ${i + 1}`)
      );
    }

    const allChats = await storage.fetchAllChats(userId, sessionId);
    expect(allChats).toHaveLength(10);
    expect(allChats.map((m) => m.content?.[0]?.text)).toEqual([
      "User message 1",
      "[agent1] Assistant message 1",
      "User message 2",
      "[agent1] Assistant message 2",
      "User message 3",
      "[agent1] Assistant message 3",
      "User message 4",
      "[agent1] Assistant message 4",
      "User message 5",
      "[agent1] Assistant message 5",
    ]);
  });

  test("fetchAllChats should respect maxMessagePairsPerAgent when set to even", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";
    const maxMessageHistory = 4; // This will allow up to 4 messages (2 pairs)

    for (let i = 0; i < 5; i++) {
      await storage.saveChatMessage(
        userId,
        sessionId,
        agentId,
        createMessage(ParticipantRole.USER, `User message ${i + 1}`),
        maxMessageHistory
      );
      await storage.saveChatMessage(
        userId,
        sessionId,
        agentId,
        createMessage(ParticipantRole.ASSISTANT, `Assistant message ${i + 1}`),
        maxMessageHistory
      );
    }

    const allChats = await storage.fetchAllChats(userId, sessionId);
    expect(allChats).toHaveLength(4);
    expect(allChats.map((m) => m.content?.[0]?.text)).toEqual([
      "User message 4",
      "[agent1] Assistant message 4",
      "User message 5",
      "[agent1] Assistant message 5",
    ]);
  });

  test("fetchAllChats should respect maxMessagePairsPerAgent when set to non-even", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";
    const maxMessageHistory = 3; // This will allow up to 4 messages (2 pairs)

    for (let i = 0; i < 4; i++) {
      await storage.saveChatMessage(
        userId,
        sessionId,
        agentId,
        createMessage(ParticipantRole.USER, `User message ${i + 1}`),
        maxMessageHistory
      );
      await storage.saveChatMessage(
        userId,
        sessionId,
        agentId,
        createMessage(ParticipantRole.ASSISTANT, `Assistant message ${i + 1}`),
        maxMessageHistory
      );
    }

    const allChats = await storage.fetchAllChats(userId, sessionId);
    expect(allChats).toHaveLength(2);
    expect(allChats.map((m) => m.content?.[0]?.text)).toEqual([
      "User message 4",
      "[agent1] Assistant message 4",
    ]);
  });

  // You might want to add another test for the case when maxMessagePairsPerAgent is not set
  test("fetchAllChats should return all messages when maxMessagePairsPerAgent is not set", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";

    for (let i = 0; i < 5; i++) {
      await storage.saveChatMessage(
        userId,
        sessionId,
        agentId,
        createMessage(ParticipantRole.USER, `User message ${i + 1}`)
      );
      await storage.saveChatMessage(
        userId,
        sessionId,
        agentId,
        createMessage(ParticipantRole.ASSISTANT, `Assistant message ${i + 1}`)
      );
    }

    const conversation = await storage.fetchChat(userId, sessionId, agentId, 2);

    expect(conversation).toHaveLength(2);

    const allChats = await storage.fetchAllChats(userId, sessionId);
    expect(allChats).toHaveLength(10);
    expect(allChats.map((m) => m.content?.[0]?.text)).toEqual([
      "User message 1",
      "[agent1] Assistant message 1",
      "User message 2",
      "[agent1] Assistant message 2",
      "User message 3",
      "[agent1] Assistant message 3",
      "User message 4",
      "[agent1] Assistant message 4",
      "User message 5",
      "[agent1] Assistant message 5",
    ]);
  });

  test("saveChatMessage should maintain message order with alternating roles", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";

    const messages = [
      createMessage(ParticipantRole.USER, "Hello"),
      createMessage(ParticipantRole.ASSISTANT, "Hi there"),
      createMessage(ParticipantRole.USER, "How are you?"),
      createMessage(ParticipantRole.ASSISTANT, "I'm doing well, thanks!"),
      createMessage(ParticipantRole.USER, "What's the weather like?"),
      createMessage(ParticipantRole.ASSISTANT, "It's sunny today."),
    ];

    for (const message of messages) {
      await storage.saveChatMessage(userId, sessionId, agentId, message);
    }

    const conversation = await storage.fetchChat(userId, sessionId, agentId);

    expect(conversation).toHaveLength(messages.length);

    for (let i = 0; i < messages.length; i++) {
      expect(conversation[i].role).toBe(messages[i].role);
      expect(conversation[i].content?.[0]?.text).toBe(
        messages[i].content?.[0]?.text
      );
    }

    // Additional check to ensure the exact order of roles
    const expectedRoleOrder = [
      ParticipantRole.USER,
      ParticipantRole.ASSISTANT,
      ParticipantRole.USER,
      ParticipantRole.ASSISTANT,
      ParticipantRole.USER,
      ParticipantRole.ASSISTANT,
    ];

    expect(conversation.map((m) => m.role)).toEqual(expectedRoleOrder);

    // Additional check to ensure the exact order of message contents
    const expectedContentOrder = [
      "Hello",
      "Hi there",
      "How are you?",
      "I'm doing well, thanks!",
      "What's the weather like?",
      "It's sunny today.",
    ];

    expect(conversation.map((m) => m.content?.[0]?.text)).toEqual(
      expectedContentOrder
    );
  });

  test('saveChatMessage should maintain order and prevent consecutive same-role messages', async () => {
    const userId = 'user1';
    const sessionId = 'session1';
    const agentId = 'agent1';

    const messages = [
      createMessage(ParticipantRole.USER, 'Hello'),
      createMessage(ParticipantRole.ASSISTANT, 'Hi there'),
      createMessage(ParticipantRole.USER, 'How are you?'),
      createMessage(ParticipantRole.USER, 'Are you there?'), // This should not be saved
      createMessage(ParticipantRole.ASSISTANT, "I'm doing well, thanks!"),
      createMessage(ParticipantRole.ASSISTANT, 'How about you?'), // This should not be saved
      createMessage(ParticipantRole.USER, "I'm good too"),
    ];

    for (const message of messages) {
      await storage.saveChatMessage(userId, sessionId, agentId, message);
    }

    const conversation = await storage.fetchChat(userId, sessionId, agentId);

    expect(conversation).toHaveLength(5);
    expect(conversation[0].content?.[0]?.text).toBe('Hello');
    expect(conversation[1].content?.[0]?.text).toBe('Hi there');
    expect(conversation[2].content?.[0]?.text).toBe('How are you?');
    expect(conversation[3].content?.[0]?.text).toBe("I'm doing well, thanks!");
    expect(conversation[4].content?.[0]?.text).toBe("I'm good too");

    // Check that roles alternate
    expect(conversation.map(m => m.role)).toEqual([
      ParticipantRole.USER,
      ParticipantRole.ASSISTANT,
      ParticipantRole.USER,
      ParticipantRole.ASSISTANT,
      ParticipantRole.USER,
    ]);
  });
 
  test('saveChatMessage should maintain correct order and handle multiple agents', async () => {
    const userId = 'user1';
    const sessionId = 'session1';
    const airlinesbotId = 'airlinesbot';
    const techAgentId = 'tech-agent';

    const createMessage = (role: ParticipantRole, text: string, agentId: string): ConversationMessage & { agentId: string } => ({
      role,
      content: [{ text }],
      agentId
    });

    const messages = [
      createMessage(ParticipantRole.USER, 'book a flight', airlinesbotId),
      createMessage(ParticipantRole.ASSISTANT, 'I see you have a frequent flyer account with us. Can you confirm ...', airlinesbotId),
      createMessage(ParticipantRole.USER, '34567', airlinesbotId),
      createMessage(ParticipantRole.ASSISTANT, 'Thank you. And for verification can I get the last four digits of...', airlinesbotId),
      createMessage(ParticipantRole.USER, '3456', airlinesbotId),
      createMessage(ParticipantRole.ASSISTANT, 'Got it. Let me get some information about your trip. Is this rese...', airlinesbotId),
      createMessage(ParticipantRole.USER, 'one way trip', airlinesbotId),
      createMessage(ParticipantRole.ASSISTANT, 'Got it. What city are you departing from?', airlinesbotId),
      createMessage(ParticipantRole.USER, 'Paris', airlinesbotId),
      createMessage(ParticipantRole.ASSISTANT, "OK. What's the total number of travelers?", airlinesbotId),
      createMessage(ParticipantRole.USER, 'what is aws lambda?', techAgentId),
      createMessage(ParticipantRole.ASSISTANT, 'AWS Lambda is a serverless computing service provided by Amazon We...', techAgentId),
      createMessage(ParticipantRole.USER, 'go back to book a flight', airlinesbotId),
      createMessage(ParticipantRole.ASSISTANT, "Paris. OK. And, what's your destination?", airlinesbotId),
      createMessage(ParticipantRole.USER, 'London', airlinesbotId),
      createMessage(ParticipantRole.ASSISTANT, 'Got it. What date would you like to take the flight?', airlinesbotId),
      createMessage(ParticipantRole.USER, 'tomorrow', airlinesbotId),
    ];

    for (const message of messages) {
      await storage.saveChatMessage(userId, sessionId, message.agentId, message);
      await delay(1);
    }

    // Fetch conversations for both agents
    const airlinesbotConversation = await storage.fetchChat(userId, sessionId, airlinesbotId);
    const techAgentConversation = await storage.fetchChat(userId, sessionId, techAgentId);

    // Check airlinesbot conversation
    expect(airlinesbotConversation.length).toBe(15); // All messages except the 2 tech-agent messages
    expect(airlinesbotConversation[airlinesbotConversation.length - 1].content?.[0]?.text).toBe('tomorrow');

    // Check tech-agent conversation
    expect(techAgentConversation.length).toBe(2);
    expect(techAgentConversation[0].content?.[0]?.text).toBe('what is aws lambda?');
    expect(techAgentConversation[1].content?.[0]?.text).toBe('AWS Lambda is a serverless computing service provided by Amazon We...');

    // Fetch all chats
    const allChats = await storage.fetchAllChats(userId, sessionId);

    // Verify total number of messages
    expect(allChats.length).toBe(messages.length);

    // Verify specific order of key messages
    expect(allChats[9].content?.[0]?.text).toBe("[airlinesbot] OK. What's the total number of travelers?");
    expect(allChats[10].content?.[0]?.text).toBe('what is aws lambda?');
    expect(allChats[11].content?.[0]?.text).toBe('[tech-agent] AWS Lambda is a serverless computing service provided by Amazon We...');
    expect(allChats[12].content?.[0]?.text).toBe('go back to book a flight');

  });

  test("fetchChat with maxHistorySize 0 or 1 should keep no messages", async () => {
    const userId = "user1";
    const sessionId = "session1";
    const agentId = "agent1";

    await storage.saveChatMessage(userId, sessionId, agentId, createMessage(ParticipantRole.USER, "Hello"));
    await storage.saveChatMessage(userId, sessionId, agentId, createMessage(ParticipantRole.ASSISTANT, "Hi there"));
    await storage.saveChatMessage(userId, sessionId, agentId, createMessage(ParticipantRole.USER, "Another question"));
    await storage.saveChatMessage(userId, sessionId, agentId, createMessage(ParticipantRole.ASSISTANT, "Another answer"));

    // slice(-0) is slice(0), so without an explicit guard these return the full history.
    expect(await storage.fetchChat(userId, sessionId, agentId, 0)).toHaveLength(0);
    expect(await storage.fetchChat(userId, sessionId, agentId, 1)).toHaveLength(0);

    // Sizes >= 2 keep the existing even-pair behaviour, and undefined keeps everything.
    const trimmed = await storage.fetchChat(userId, sessionId, agentId, 2);
    expect(trimmed).toHaveLength(2);
    expect(trimmed[1].content?.[0]?.text).toBe("Another answer");
    expect(await storage.fetchChat(userId, sessionId, agentId)).toHaveLength(4);
  });


});
