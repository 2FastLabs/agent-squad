# Natural Language to Structured Data

A demonstration of how to transform free-text queries into structured, actionable data using a multi-agent architecture.

## Overview

This project implements a proof-of-concept system that:
1. Takes natural language input from users
2. Routes queries to specialized agents using an orchestrator
3. Transforms free text into structured formats (JSON for product searches, contextual responses for returns)

Perfect for teams looking to build systems that need to:
- Convert customer queries into structured database searches
- Transform natural language into API-ready parameters
- Handle multiple types of queries with different output structures
- Maintain context and provide real-time responses

## Overview

This project implements a multi-agent orchestration system designed to handle various e-commerce related queries. It features:

- **Product Search Agent**: Processes natural language queries about products and converts them into structured search parameters
- **Returns & Terms Assistant**: Handles inquiries about return policies, refunds, and terms & conditions
- **Greeting Agent**: Welcomes users and helps them navigate the available services

The system uses AWS Bedrock with Haiku 4.5 as the underlying language model and implements streaming responses for real-time interaction.

## Prerequisites

- Python 3.12
- AWS Account with Bedrock access

## Setup

Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script using:

```bash
python main.py
```

The system will start in interactive mode, allowing you to input queries and receive responses in real-time.

## Sample Interactions

### 1. Product Search

**Input:**
```
Show me Prime-eligible headphones under $100 with good reviews
```

**Output:**
```json
{
  "department": "Electronics",
  "categories": ["Headphones"],
  "priceRange": {
    "max": 100,
    "currency": "USD"
  },
  "customerReview": {
    "stars": 4,
    "operator": "gte"
  },
  "shippingOptions": {
    "prime": true
  },
  "condition": "New"
}
```

### 2. Returns Policy

**Input:**
```
How do I return a damaged item?
```

**Output:**
```
For damaged items, here's our return process:

1. Document the damage with photos
2. Contact customer service within 48 hours of receipt
3. You'll receive a prepaid return label
4. Package the item securely with all original materials
5. Ship within 14 days of receiving the return label

The refund will be processed to your original payment method within 3-5 business days after we receive the item. Shipping costs are covered for damaged items.
```

### 3. General Greeting

**Input:**
```
hello
```

**Output:**
```markdown
## Welcome! 👋

I'm the greeting agent, here to help you navigate our services. We have several specialized agents available:

- **Product Search Agent**: Find products, compare prices, and discover deals
- **Returns and Terms Assistant**: Get help with returns, refunds, and policies

How can we assist you today? Feel free to ask about:
- Product searches and recommendations
- Return policies and procedures
- General assistance and guidance
```

## Agents

The system is built on three main components:

1. **AgentSquad**: Routes queries to appropriate agents
2. **Agents**: Specialized handlers for different types of queries
3. **Streaming Handler**: Manages real-time response generation


### Product Search Agent
The current implementation demonstrates the agent's capability to convert natural language queries into structured JSON output. This is just the first step - in a production environment, you would:

1. Implement the TODO section in the `process_request` method
2. Add calls to your internal APIs, databases, or search engines
3. Use the structured JSON to query your product catalog
4. Return actual product results instead of just the parsed query

Example implementation in the TODO section:
```python
# After getting parsed_response:
products = await your_product_service.search(
    department=parsed_response['department'],
    price_range=parsed_response['priceRange'],
    # ... other parameters
)
return ConversationMessage(
    role=ParticipantRole.ASSISTANT.value,
    content=[{"text": format_product_results(products)}]
)
```

### Returns and Terms Assistant
The current implementation uses a static prompt. To make it more powerful and maintenance-friendly:

1. Integrate with a vector storage solution like [Amazon Bedrock Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) or other vector databases
2. Set up a retrieval system to fetch relevant policy documents
3. Update the agent's prompt with retrieved context

Example enhancement:
```python
retriever = BedrockKnowledgeBaseRetriever(
    kb_id="your-kb-id",
    region_name="your-region"
)
# Add to the agent's configuration
```

### Greeting Agent
The greeting agent has been implemented as a crucial component for chat-based interfaces. Its primary purposes are:

1. Providing a friendly entry point to the system
2. Helping users understand available capabilities
3. Guiding users toward the most appropriate agent
4. Reducing user confusion and improving engagement

This pattern is especially useful in chat interfaces where users might not initially know what kinds of questions they can ask or which agent would best serve their needs.

