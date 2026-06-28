import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: process.env.ASTRO_SITE,
	base: '/agent-squad',
	markdown: {
		gfm: true
  },
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
			social: {
				github: 'https://github.com/2fastlabs/agent-squad'
			},
			sidebar: [
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
						label: 'Agentic Patterns',
						items: [
						  { label: 'Agentic Patterns Overview', link: '/cookbook/patterns/agentic-patterns' },
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
			  ]
		})
	]
});
