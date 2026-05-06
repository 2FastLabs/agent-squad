import {
  DynamoDBDocumentClient,
  PutCommand,
  GetCommand,
  QueryCommand,
} from "@aws-sdk/lib-dynamodb";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { ChatStorage } from "./chatStorage";
import { ConversationMessage, ParticipantRole, TimestampedMessage } from "../types";
import { Logger } from "../utils/logger";



export class DynamoDbChatStorage extends ChatStorage {
  private tableName: string;
  private docClient: DynamoDBDocumentClient;
  private ttlKey: string | null = null;
  private ttlDuration: number = 3600;

  constructor(tableName: string, region: string, ttlKey?: string, ttlDuration?: number) {
    super();
    this.tableName = tableName;
    this.ttlKey = ttlKey || null;
    this.ttlDuration = Number(ttlDuration) || 3600;
    const client = new DynamoDBClient({ region });
    this.docClient = DynamoDBDocumentClient.from(client);
  }

  /**
   * Allows subclasses to customize the DynamoDB Document Client.
   * Useful for local development, testing, or custom endpoint configurations.
   * @param client - The DynamoDBDocumentClient to use
   * @protected
   */
  protected setDocClient(client: DynamoDBDocumentClient): void {
    this.docClient = client;
  }

  async saveChatMessage(
    userId: string,
    sessionId: string,
    agentId: string,
    newMessage: ConversationMessage,
    maxHistorySize?: number
  ): Promise<ConversationMessage[]> {
    const key = this.generateKey(userId, sessionId, agentId);
    // Fetch existing conversation
    const existingConversation = await this.fetchChat(userId, sessionId, agentId);
    
    if (super.isConsecutiveMessage(existingConversation, newMessage)) {
      Logger.logger.log(`> Consecutive ${newMessage.role} message detected for agent ${agentId}. Not saving.`);
      return existingConversation;
    }

    // Add new message with timestamp
    const updatedConversation: TimestampedMessage[] = [
      ...existingConversation.map(msg => ({ ...msg, timestamp: Date.now() })),
      { ...newMessage, timestamp: Date.now() }
    ];

    // Apply maxHistorySize limit if specified
    const trimmedConversation = super.trimConversation(updatedConversation, maxHistorySize);

    // Prepare item for DynamoDB
    const item: Record<string, any> = {
      PK: userId,
      SK: key,
      conversation: trimmedConversation,
    };

    if (this.ttlKey) {
      item[this.ttlKey] = Math.floor(Date.now() / 1000) + this.ttlDuration;
    }

    // Save to DynamoDB
    try {
      await this.docClient.send(new PutCommand({
        TableName: this.tableName,
        Item: item,
      }));
    } catch (error) {
      Logger.logger.error("Error saving conversation to DynamoDB:", error);
      throw error;
    }

    // Return the updated conversation without timestamps
    return trimmedConversation;
  }

  async fetchChat(
    userId: string,
    sessionId: string,
    agentId: string
  ): Promise<ConversationMessage[]> {
    const key = this.generateKey(userId, sessionId, agentId);
    try {
      const response = await this.docClient.send(new GetCommand({
        TableName: this.tableName,
        Key: { PK: userId, SK: key },
      }));
      const storedMessages: TimestampedMessage[] = response.Item?.conversation || [];

      return this.removeTimestamps(storedMessages);
    } catch (error) {
      Logger.logger.error("Error getting conversation from DynamoDB:", error);
      throw error;
    }
  }

  
  async fetchAllChats(userId: string, sessionId: string): Promise<ConversationMessage[]> {
    try {
      const response = await this.docClient.send(new QueryCommand({
        TableName: this.tableName,
        KeyConditionExpression: "PK = :pk and begins_with(SK, :skPrefix)",
        ExpressionAttributeValues: {
          ":pk": userId,
          ":skPrefix": `${sessionId}#`,
        },
      }));
  
      if (!response.Items || response.Items.length === 0) {
        return [];
      }
  
      const allChats = response.Items.flatMap(item => {
        if (!Array.isArray(item.conversation)) {
          Logger.logger.error("Unexpected item structure:", item);
          return [];
        }
  
        // Extract agentId from the SK
        const agentId = item.SK.split('#')[1];
  
        return item.conversation.map(msg => ({
          role: msg.role,
          content: msg.role === ParticipantRole.ASSISTANT
            ? [{ text: `[${agentId}] ${Array.isArray(msg.content) ? msg.content[0]?.text || '' : msg.content || ''}` }]
            : (Array.isArray(msg.content) ? msg.content.map(content => ({ text: content.text })) : [{ text: msg.content || '' }]),
          timestamp: Number(msg.timestamp)
        } as TimestampedMessage));
      });
  
      allChats.sort((a, b) => a.timestamp - b.timestamp);
      return this.removeTimestamps(allChats);
    } catch (error) {
      Logger.logger.error("Error querying conversations from DynamoDB:", error);
      throw error;
    }
  }


  private generateKey(userId: string, sessionId: string, agentId: string): string {
    return `${sessionId}#${agentId}`;
  }

  private removeTimestamps(messages: TimestampedMessage[] | ConversationMessage[]): ConversationMessage[] {
    return messages.map(msg => {
      const { timestamp:_timestamp, ...rest } = msg as TimestampedMessage;
      return rest;
    });
  }
}