import { Transform, TransformCallback } from 'stream';
import { ConversationMessage, ToolInput } from '../types';


export class AccumulatorTransform extends Transform {
    private accumulator: string;

    constructor() {
      super({
        objectMode: true  // This allows the transform to handle object chunks
      });
      this.accumulator = '';
    }

    _transform(chunk: any, encoding: string, callback: TransformCallback): void {
      // A widget chunk is forwarded to the consumer but never folded into the accumulated text
      // answer (which is what gets saved to storage).
      if (chunk && typeof chunk === 'object' && chunk.ui) {
        this.push(chunk);
        callback();
        return;
      }
      const text = this.extractFromChunk(chunk);
      if (text) {
        this.accumulator += text;
        this.push(text);  // Push the text, not the original chunk
      }
      callback();
    }

    extractFromChunk(chunk: any): string | null | any {
      if (typeof chunk === 'string') {
        return chunk;
      } else if (chunk.contentBlockDelta?.delta?.text) {
        return chunk.contentBlockDelta.delta.text;
      } else if (chunk.thinking) {
        return chunk;
      }
      // Add more conditions here if there are other possible structures
      return null;
    }

    getAccumulatedData(): string {
      return this.accumulator;
    }
  }

  export function extractXML(text: string) {
    const xmlRegex = /<response>[\s\S]*?<\/response>/;
    const match = text.match(xmlRegex);
    return match ? match[0] : null;
  }


  export function isClassifierToolInput(input: unknown): input is ToolInput {
    return (
      typeof input === 'object' &&
      input !== null &&
      'userinput' in input &&
      'selected_agent' in input &&
      'confidence' in input
    );
  }

  export function isConversationMessage(result: any): result is ConversationMessage {
    return (
      result &&
      typeof result === "object" &&
      "role" in result &&
      "content" in result &&
      Array.isArray(result.content)
    );
  }


