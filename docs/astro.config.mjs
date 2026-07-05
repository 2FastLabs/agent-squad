import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightSidebarTopics from 'starlight-sidebar-topics';
import starlightThemeRapide from 'starlight-theme-rapide';

// https://astro.build/config
export default defineConfig({
	site: process.env.ASTRO_SITE,
	base: '/agent-squad',
	integrations: [
		starlight({
			title: 'Agent Squad',
			description: 'Flexible and powerful framework for managing multiple AI agents and handling complex conversations 🤖🚀',
			defaultLocale: 'en',
			favicon: '/src/assets/favicon.ico',
			customCss: [
				'./src/styles/landing.css',
				'./src/styles/font.css',
				'./src/styles/custom.css',
				'./src/styles/terminal.css'
			],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/2fastlabs/agent-squad' },
			],
			plugins: [
				starlightThemeRapide(),
				starlightSidebarTopics([
					{
						label: 'Python / TypeScript',
						link: '/general/introduction',
						items: [
							{
							  label: 'Introduction',
							  items: [
								{ label: 'Introduction', link: '/general/introduction' },
								{ label: 'How it works', link: '/general/how-it-works' },
								{ label: 'Quickstart', link: '/general/quickstart' },
								{ label: 'FAQ', link: '/general/faq' }
							  ]
							},
							{
								label: 'Orchestrator',
								items: [
								  { label: 'Overview', link: '/orchestrator/overview' },
								]
							},{
								label: 'Classifier',
								items: [
								  { label: 'Overview', link: '/classifiers/overview' },
								  {
									label: 'Built-in classifiers',
									items: [
									  { label: 'Bedrock Classifier', link: '/classifiers/built-in/bedrock-classifier'},
									  { label: 'Anthropic Classifier', link: '/classifiers/built-in/anthropic-classifier' },
									  { label: 'OpenAI Classifier', link: '/classifiers/built-in/openai-classifier' },
									]
								  },
								  { label: 'Custom Classifier', link: '/classifiers/custom-classifier' },

								]
							},
							{
							  label: 'Agents',
							  items: [
								{ label: 'Overview', link: '/agents/overview' },
								{
								  label: 'Built-in Agents',
								  items: [
									{ label: 'Supervisor Agent', link: '/agents/built-in/supervisor-agent' },
									{ label: 'Bedrock LLM Agent', link: '/agents/built-in/bedrock-llm-agent'},
									{ label: 'Amazon Bedrock Agent', link: '/agents/built-in/amazon-bedrock-agent' },
									{ label: 'Amazon Lex Bot Agent', link: '/agents/built-in/lex-bot-agent' },
									{ label: 'AWS Lambda Agent', link: '/agents/built-in/lambda-agent' },
									{ label: 'OpenAI Agent', link: '/agents/built-in/openai-agent' },
									{ label: 'Anthropic Agent', link: '/agents/built-in/anthropic-agent'},
									{ label: 'Chain Agent', link: '/agents/built-in/chain-agent' },
									{ label: 'Grounded Agent', link: '/agents/built-in/grounded-agent' },
									{ label: 'Comprehend Filter Agent', link: '/agents/built-in/comprehend-filter-agent' },
									{ label: 'Amazon Bedrock Translator Agent', link: '/agents/built-in/bedrock-translator-agent' },
									{ label: 'Amazon Bedrock Inline Agent', link: '/agents/built-in/bedrock-inline-agent' },
									{ label: 'Bedrock Flows Agent', link: '/agents/built-in/bedrock-flows-agent' },
								  ]
								},
								{ label: 'Custom Agents', link: '/agents/custom-agents' },
								{ label: 'Tools for Agents', link: '/agents/tools' },

							  ]
							},
							{
							  label: 'Conversation Storage',
							  items: [
								{ label: 'Overview', link: '/storage/overview' },
								{
									label: 'Built-in storage',
									items: [
										{ label: 'In-Memory', link: '/storage/in-memory' },
										{ label: 'DynamoDB', link: '/storage/dynamodb' },
										{ label: 'SQL Storage', link: '/storage/sql' },
									]
								},
								{ label: 'Custom Storage', link: '/storage/custom' }
							  ]
							},
							{
								label: 'Retrievers',
								items: [
								  { label: 'Overview', link: '/retrievers/overview' },
								  {
									label: 'Built-in retrievers',
									items: [
										{ label: 'Bedrock Knowledge Base', link: '/retrievers/built-in/bedrock-kb-retriever' },
									]
								},
								  { label: 'Custom Retriever', link: '/retrievers/custom-retriever' },
								]
							},
							{
								label: 'Cookbook',
								items: [
								  {
									label: 'Examples',
									items: [
									  { label: 'Chat Chainlit App', link: '/cookbook/examples/chat-chainlit-app' },
									  { label: 'Chat Demo App', link: '/cookbook/examples/chat-demo-app' },
									  { label: 'E-commerce Support Simulator', link: '/cookbook/examples/ecommerce-support-simulator' },
									  { label: 'Fast API Streaming', link: '/cookbook/examples/fast-api-streaming' },
									  { label: 'Typescript Local Demo', link: '/cookbook/examples/typescript-local-demo' },
									  { label: 'Python Local Demo', link: '/cookbook/examples/python-local-demo' },
									  { label: 'Api Agent', link: '/cookbook/examples/api-agent' },
									  { label: 'Ollama Agent', link: '/cookbook/examples/ollama-agent' },
									  { label: 'Ollama Classifier', link: '/cookbook/examples/ollama-classifier' }
									]
								  },
								  {
									label: 'Lambda Implementations',
									items: [
									  { label: 'Python Lambda', link: '/cookbook/lambda/aws-lambda-python' },
									  { label: 'NodeJs Lambda', link: '/cookbook/lambda/aws-lambda-nodejs' }
									]
								  },
								  {
									label: 'Tool Integration',
									items: [
									  { label: 'Weather API Integration', link: '/cookbook/tools/weather-api' },
									  { label: 'Math Operations', link: '/cookbook/tools/math-operations' }
									]
								  },
								  {
									label: 'Routing Patterns',
									items: [
									  { label: 'Cost-Efficient Routing', link: '/cookbook/patterns/cost-efficient' },
									  { label: 'Multi-lingual Routing', link: '/cookbook/patterns/multi-lingual' }
									]
								  },
								  {
									label: 'Optimization, Logging & Observability',
									items: [
									  { label: 'Agent Overlap Analysis', link: '/cookbook/monitoring/agent-overlap' },
									  { label: 'Logging', link: '/cookbook/monitoring/logging' },
									  { label: 'Observability', link: '/cookbook/monitoring/observability' }
									]
								  }
								]
							  }
						],
					},
					{
						label: 'Swift',
						link: '/swift/quick-start',
						items: [
							{
								label: 'Start here',
								items: [
									{ label: 'Quick start', slug: 'swift/quick-start' },
									{ label: 'Extending the framework', slug: 'swift/guides/extending' },
									{ label: 'Building with an AI assistant', slug: 'swift/guides/building-with-ai' },
								],
							},
							{
								label: 'Orchestrator',
								items: [{ label: 'Overview', slug: 'swift/orchestrator/overview' }],
							},
							{
								label: 'Agents',
								items: [
									{ label: 'Overview', slug: 'swift/agents/overview' },
									{
										label: 'Built-in agents',
										items: [
											{ label: 'Agent', slug: 'swift/agents/built-in/agent' },
											{ label: 'GroundedAgent', slug: 'swift/agents/built-in/grounded-agent' },
										],
									},
									{ label: 'Custom agent', slug: 'swift/agents/custom' },
								],
							},
							{
								label: 'Classifiers',
								items: [
									{ label: 'Overview', slug: 'swift/classifiers/overview' },
									{
										label: 'Built-in classifiers',
										items: [{ label: 'LLMClassifier', slug: 'swift/classifiers/built-in/llm-classifier' }],
									},
									{ label: 'Custom classifier', slug: 'swift/classifiers/custom' },
								],
							},
							{
								label: 'Tools',
								items: [
									{ label: 'Overview', slug: 'swift/tools/overview' },
									{
										label: 'Built-in tools',
										items: [
											{ label: 'Local & HTTP tools', slug: 'swift/tools/built-in/local-http' },
											{ label: 'Composing providers', slug: 'swift/tools/built-in/composing' },
											{ label: 'MCP servers', slug: 'swift/mcp/overview' },
											{ label: 'MCP client (SDKMCPClient)', slug: 'swift/mcp/built-in/sdk-client' },
										],
									},
									{
										label: 'Tool UIs',
										items: [
											{ label: 'Overview', slug: 'swift/ui/overview' },
											{ label: 'Curators & PresenterPrompt', slug: 'swift/ui/built-in/curators' },
											{ label: 'Custom curator', slug: 'swift/ui/custom' },
										],
									},
									{ label: 'Custom tool provider', slug: 'swift/tools/custom' },
									{ label: 'Custom MCP client', slug: 'swift/mcp/custom' },
								],
							},
							{
								label: 'Chat history',
								items: [
									{ label: 'Overview', slug: 'swift/storage/overview' },
									{
										label: 'Built-in stores',
										items: [
											{ label: 'In-memory', slug: 'swift/storage/built-in/in-memory' },
											{ label: 'File', slug: 'swift/storage/built-in/file' },
											{ label: 'Device (SwiftData)', slug: 'swift/storage/built-in/device' },
										],
									},
									{ label: 'Custom store', slug: 'swift/storage/custom' },
								],
							},
							{
								label: 'LLM clients',
								items: [
									{ label: 'Overview', slug: 'swift/llm/overview' },
									{
										label: 'Built-in clients',
										items: [{ label: 'ChatCompletionsClient', slug: 'swift/llm/built-in/chat-completions' }],
									},
									{ label: 'Custom connector', slug: 'swift/llm/custom' },
								],
							},
							{
								label: 'Tracing',
								items: [
									{ label: 'Overview', slug: 'swift/tracing/overview' },
									{
										label: 'Built-in tracing',
										items: [
											{ label: 'OSLogTracer', slug: 'swift/tracing/built-in/oslog-tracer' },
											{ label: 'ProcessingTracer', slug: 'swift/tracing/built-in/processing-tracer' },
											{ label: 'OTLP exporter', slug: 'swift/tracing/built-in/otlp-exporter' },
											{ label: 'BatchSpanProcessor', slug: 'swift/tracing/built-in/batch-span-processor' },
										],
									},
									{ label: 'Custom tracing', slug: 'swift/tracing/custom' },
								],
							},
							{
								label: 'Realtime voice',
								items: [
									{ label: 'Overview', slug: 'swift/voice/overview' },
									{
										label: 'Built-in voice',
										items: [
											{ label: 'OpenAIVoiceAssistant', slug: 'swift/voice/built-in/openai-voice' },
											{ label: 'OpenAIGroundedVoiceAssistant', slug: 'swift/voice/built-in/openai-grounded-voice' },
											{ label: 'WebSocket transport', slug: 'swift/voice/built-in/websocket-transport' },
										],
									},
									{
										label: 'Audio',
										items: [
											{ label: 'Overview', slug: 'swift/audio/overview' },
											{ label: 'VoiceProcessedAudioIO', slug: 'swift/audio/built-in/voice-processed-audio-io' },
											{ label: 'MicCapture', slug: 'swift/audio/built-in/mic-capture' },
											{ label: 'AudioPlayback', slug: 'swift/audio/built-in/audio-playback' },
											{ label: 'Custom audio', slug: 'swift/audio/custom' },
										],
									},
									{ label: 'Custom transport', slug: 'swift/voice/custom' },
								],
							},
							{
								label: 'Examples',
								items: [
									{ label: 'Overview', slug: 'swift/examples/overview' },
									{ label: 'Local tool', slug: 'swift/examples/local-tool' },
									{ label: 'API tools', slug: 'swift/examples/api-tools' },
									{ label: 'MCP server', slug: 'swift/examples/mcp-server' },
								],
							},
							{
								label: 'Reference',
								items: [{ label: 'Messages & events', slug: 'swift/reference/messages-and-events' }],
							},
						],
					},
				]),
			],
		})
	]
});
