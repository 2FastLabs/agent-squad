import { DynamoDbChatStorage } from "../../src/storage/dynamoDbChatStorage";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand, GetCommand } from "@aws-sdk/lib-dynamodb";
import { ConversationMessage, ParticipantRole } from "../../src/types";

// Mock AWS SDK
jest.mock("@aws-sdk/client-dynamodb");
jest.mock("@aws-sdk/lib-dynamodb", () => ({
  DynamoDBDocumentClient: {
    from: jest.fn(),
  },
  PutCommand: jest.fn(),
  GetCommand: jest.fn(),
  QueryCommand: jest.fn(),
}));

describe("DynamoDbChatStorage", () => {
  let storage: DynamoDbChatStorage;
  let mockDocClient: jest.Mocked<DynamoDBDocumentClient>;
  
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Create mock document client
    mockDocClient = {
      send: jest.fn().mockResolvedValue({ Item: { conversation: [] } }),
    } as any;
    
    // Mock DynamoDBDocumentClient.from to return our mock
    (DynamoDBDocumentClient.from as jest.Mock).mockReturnValue(mockDocClient);
    
    // Create storage instance
    storage = new DynamoDbChatStorage("test-table", "us-east-1", "ttl", 3600);
  });

  describe("setDocClient", () => {
    it("should allow subclasses to override the document client", async () => {
      // Create a mock custom document client
      const customDocClient = {
        send: jest.fn().mockResolvedValue({ Item: { conversation: [] } }),
      } as any;
      
      // Create a test subclass that uses setDocClient
      class TestDynamoDbChatStorage extends DynamoDbChatStorage {
        constructor(tableName: string, region: string) {
          super(tableName, region);
          // Use the protected method to set custom client
          this.setDocClient(customDocClient);
        }
      }
      
      // Create instance of subclass
      const testStorage = new TestDynamoDbChatStorage("test-table", "us-east-1");
      
      // Call a method that uses the document client
      await testStorage.fetchChat("user1", "session1", "agent1");
      
      // Verify that the custom client was used, not the original mock
      expect(customDocClient.send).toHaveBeenCalled();
      expect(mockDocClient.send).not.toHaveBeenCalled();
    });

    it("should work correctly for local development scenario", async () => {
      // Mock a local document client
      const localDocClient = {
        send: jest.fn()
          .mockResolvedValueOnce({ Item: { conversation: [] } }) // for fetchChat
          .mockResolvedValueOnce({}), // for saveChatMessage
      } as any;
      
      // Override the mock to return local client for the second call
      let callCount = 0;
      (DynamoDBDocumentClient.from as jest.Mock).mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return mockDocClient; // First call in parent constructor
        }
        return localDocClient; // Second call in subclass
      });
      
      // Simulate local development setup
      class LocalDynamoDbChatStorage extends DynamoDbChatStorage {
        constructor(tableName: string, region: string, endpoint: string) {
          super(tableName, region);
          
          const client = new DynamoDBClient({
            region,
            endpoint,
            credentials: {
              accessKeyId: "dummy",
              secretAccessKey: "dummy",
            },
          });
          
          this.setDocClient(DynamoDBDocumentClient.from(client));
        }
      }
      
      const localStorage = new LocalDynamoDbChatStorage(
        "local-table",
        "us-east-1",
        "http://localhost:8000"
      );
      
      // Create and save a message
      const message: ConversationMessage = {
        role: ParticipantRole.USER,
        content: [{ text: "Hello" }],
      };
      
      // This should use the custom client with local endpoint
      const result = await localStorage.saveChatMessage("user1", "session1", "agent1", message);
      
      // Verify the local client was called
      expect(localDocClient.send).toHaveBeenCalledTimes(2); // Once for fetchChat, once for putCommand
      expect(mockDocClient.send).not.toHaveBeenCalled();
      expect(result).toHaveLength(1);
    });
  });
});