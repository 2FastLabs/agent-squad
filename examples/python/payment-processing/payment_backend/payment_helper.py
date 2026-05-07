import json
from typing import List, Dict, Any
from multi_agent_orchestrator.retrievers import Retriever

class PaymentHelper(Retriever):
    def __init__(self, data_file: str):
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.workers = json.load(f)

    async def retrieve(self, worker_id: str) -> Dict[str, Any]:
        for worker in self.workers:
            if worker["worker_id"] == worker_id:
                return worker
        return {}

    async def retrieve_and_combine_results(self, worker_id: str) -> str:
        worker_data = await self.retrieve(worker_id)
        return json.dumps(worker_data)

    async def retrieve_and_generate(self, worker_id: str) -> str:
        return await self.retrieve_and_combine_results(worker_id)

    def validate_payment_request(self, worker_id: str, payment_amount: float) -> str:
        worker = next((w for w in self.workers if w["worker_id"] == worker_id), None)
        
        if not worker:
            return json.dumps({"status": "failed", "message": "Worker not found."})
        
        if not worker["payment_history"]:
            return json.dumps({"status": "failed", "message": "Insufficient payment history."})
        
        if payment_amount > 1000:
            return json.dumps({"status": "failed", "message": "Payment amount exceeds limit."})
        
        return json.dumps({"status": "success", "message": "Payment request validated."})

    def detect_fraud(self, worker_id: str, payment_amount: float, device_id: str, location_id: str) -> str:
        worker = next((w for w in self.workers if w["worker_id"] == worker_id), None)
        
        if not worker:
            return json.dumps({"status": "failed", "message": "Worker not found."})
               
        payment_history = worker.get("payment_history", [])
        
        if not payment_history:
            return json.dumps({"status": "failed", "message": "Fraud detected: No payment history."})
             
        avg_payment = sum(payment_history) / len(payment_history)
        
        if payment_amount > 2 * avg_payment:
            return json.dumps({"status": "failed", "message": "Fraud detected: Payment amount is unusually high."})
        
        if location_id.lower() not in worker.get("location", "").lower():
            return json.dumps({"status": "failed", "message": "Fraud detected: Suspicious location."})
        
        if device_id not in worker.get("registered_devices", []):
            return json.dumps({"status": "failed", "message": "Fraud detected: Device ID not registered with this worker."})
        
        return json.dumps({"status": "success", "message": "No fraud detected."})

    def issue_payment(self, worker_id: str, payment_amount: float) -> str:
        worker = next((w for w in self.workers if w["worker_id"] == worker_id), None)
        # update workers.json with new payment
        if not worker:
            return json.dumps({"status": "failed", "message": "Worker not found."})
        
        worker["payment_history"].append(payment_amount)
        with open(self.data_file, 'w') as f:
            json.dump(self.workers, f)
        
        return json.dumps({"status": "success", "message": f"Payment of {payment_amount} issued to worker {worker_id}"})