import { ConversationMessage } from "../types";
import { AgentSquadConfig } from "../orchestrator";
import { ClassifierResult } from "../classifiers/classifier";


export class Logger {

    static logger: any | Console = console;
    private config: Partial<AgentSquadConfig>;
  constructor(config: Partial<AgentSquadConfig>= {}, logger:any = console) {
    this.config = config;
    this.setLogger(logger);
  }

  private setLogger(logger: any): void {
    Logger.logger = logger;
  }

  public info(message: string, ...params: any[]): void {
    Logger.logger.info(message, ...params);
  }

  public warn(message: string, ...params: any[]): void {
    Logger.logger.warn(message, ...params);
  }

  public error(message: string, ...params: any[]): void {
    Logger.logger.error(message, ...params);
  }

  public debug(message: string, ...params: any[]): void {
    Logger.logger.debug(message, ...params);
  }

  public log(message: string, ...params: any[]): void {
    Logger.logger.log(message, ...params);
  }

  private logHeader(title: string): void {
    Logger.logger.info(`\n** ${title.toUpperCase()} **`);
    Logger.logger.info('='.repeat(title.length + 6));
  }

  printChatHistory(chatHistory: ConversationMessage[], agentId: string | null = null): void {
    const isAgentChat = agentId !== null;
    if (isAgentChat && !this.config.LOG_AGENT_CHAT) return;
    if (!isAgentChat && !this.config.LOG_CLASSIFIER_CHAT) return;

    const title = isAgentChat
      ? `Agent ${agentId} Chat History`
      : 'Classifier Chat History';
    this.logHeader(title);

    if (chatHistory.length === 0) {
      Logger.logger.info('> - None -');
    } else {
      chatHistory.forEach((message, index) => {
        const role = message.role?.toUpperCase() ?? 'UNKNOWN';
        const content = Array.isArray(message.content) ? message.content[0] : message.content;
        const text = typeof content === 'string' ? content : content?.text ?? '';
        const trimmedText = text.length > 80 ? `${text.slice(0, 80)}...` : text;
        Logger.logger.info(`> ${index + 1}. ${role}:${trimmedText}`);
      });
    }

    this.info('');
  }

  logClassifierOutput(output: any, isRaw: boolean = false): void {
    if (isRaw && !this.config.LOG_CLASSIFIER_RAW_OUTPUT) return;
    if (!isRaw && !this.config.LOG_CLASSIFIER_OUTPUT) return;

    this.logHeader(isRaw ? 'Raw Classifier Output' : 'Processed Classifier Output');
    if (isRaw) Logger.logger.info(output);
    else Logger.logger.info(JSON.stringify(output, null, 2));
    Logger.logger.info('');
  }

  printIntent(userInput: string, intentClassifierResult: ClassifierResult): void {
    if (!this.config.LOG_CLASSIFIER_OUTPUT) return;

    this.logHeader('Classified Intent');

    Logger.logger.info(`> Text: ${userInput}`);
    Logger.logger.info(`> Selected Agent: ${
      intentClassifierResult.selectedAgent
        ? intentClassifierResult.selectedAgent.name
        : "No agent selected"
    }`);
    Logger.logger.info(`> Confidence: ${
      (intentClassifierResult.confidence || 0).toFixed(2)
    }`);

    this.info('');
  }

  printExecutionTimes(executionTimes: Map<string, number>): void {
    if (!this.config.LOG_EXECUTION_TIMES) return;

    this.logHeader('Execution Times');

    if (executionTimes.size === 0) {
      Logger.logger.info('> - None -');
    } else {
      executionTimes.forEach((duration, timerName) => {
        Logger.logger.info(`> ${timerName}: ${duration}ms`);
      });
    }

    Logger.logger.info('');
  }
}