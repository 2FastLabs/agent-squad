export const BEDROCK_MODEL_ID_CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0";
export const BEDROCK_MODEL_ID_CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0";
export const BEDROCK_MODEL_ID_CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0";
export const BEDROCK_MODEL_ID_LLAMA_3_70B = "meta.llama3-70b-instruct-v1:0";
export const OPENAI_MODEL_ID_GPT_O_MINI = "gpt-4o-mini";
export const ANTHROPIC_MODEL_ID_CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620";
export const ANTHROPIC_MODEL_ID_CLAUDE_SONNET_4 = "claude-sonnet-4-20250514";

export const AgentTypes = {
  DEFAULT: "Common Knowledge",
  CLASSIFIER : "classifier",
} as const;

export type AgentTypes = typeof AgentTypes[keyof typeof AgentTypes];

export interface ToolInput {
  userinput: string;
  selected_agent: string;
  confidence: string;
}

/**
 * Represents a streaming response that can be asynchronously iterated over.
 * This type is useful for handling responses that come in chunks or streams.
 */
export type StreamingResponse = {
  [Symbol.asyncIterator]: () => AsyncIterator<string, void, unknown>;
};


/**
 * Represents the possible roles in a conversation.
 */
export enum ParticipantRole {
  ASSISTANT = "assistant",
  USER = "user"
}

/**
 * Represents a single message in a conversation, including its content and the role of the sender.
 */
export interface ConversationMessage {
  role: ParticipantRole;
  content: any[] | undefined;
}

/**
 * Extends the Message type with a timestamp.
 * This is useful for tracking when messages were created or modified.
 */
export type TimestampedMessage = ConversationMessage & { timestamp: number };

export interface TemplateVariables {
  [key: string]: string | string[];
}


export enum AgentProviderType {
  BEDROCK = "BEDROCK",
  ANTHROPIC = "ANTHROPIC"
}

