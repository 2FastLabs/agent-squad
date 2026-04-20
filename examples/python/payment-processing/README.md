# Multi-Agent Payment Processing System

A Streamlit-based application that processes payments for gig workers using a multi-agent orchestration system. The system performs validation, fraud detection, and payment processing through a chain of specialized agents.

## Features

- Payment request validation through a multi-agent system
- Demonistration of building building a graph or chaining multiple agents
- Agents using deterministic function calling as tools to make decision
- Fraud detection analysis using historical data
- Secure payment processing with device and location verification
- User-friendly Streamlit web interface for initiating payments

## System Architecture

The application uses three main agents orchestrated in a chain:

1. **Validation Agent**: Verifies worker eligibility for payments
2. **Fraud Detection Agent**: Analyzes transactions for suspicious patterns
3. **Payment Agent**: Handles the actual payment processing

## Prerequisites

- Python 3.x
- Streamlit
- Amazon Bedrock access
- Required Python packages:
  - uuid
  - asyncio
  - json
  - streamlit
  - multi_agent_orchestrator
## Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system requires a `workers.json` file in the `payment_backend` directory containing worker information with the following structure:

```json
{
  "worker_id": string,
  "payment_history": [float],
  "registered_devices": [string],
  "location": string
}
```

The `workers.json` file should be placed in the `payment_backend` subdirectory of the project.

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Enter the required information in the web interface (check details in `workers.json` file under `payment_backend` folder):
   - Worker ID
   - Payment Amount
   - Device ID
   - Location

3. Click "Process Payment" to initiate the payment process

4. Enable Bedrock [Logging](https://community.aws/content/2kkmm6q5ae1AruqcgHADjNl1Zx0/monitoring-foundation-models-with-amazon-cloudwatch) and you can see individual agent's messages passing to other sequentially with their analysis and tool usage in CloudWatch Logs.

### How it Works?
- To test successful payments use worker id: `12345`, device id: `device123`, payment amount: `500` and location: `New York`
- To test a invalid payment request, change the amount to greater than `2000`
- If the payment is successful, it will update the [workers.json](./payment_backend/workers.json)
- If the requested pyment is 2 times the average payment history, it will flag as fraud. for example, worker id: `67890`, device id: `device456`, payment amount: `135` and location: `Seattle`. as the average payment history for this worker is $65 and requested amount is >130 (2 times average) as per the `detect_fraud` method used by the agent.
- Similarly you can check other failed check such as wrong location, wrong device id, invalid worker id which are not found in the backend `worker.json`.
- Note that for each successful transaction the file gets updated with new payment history.