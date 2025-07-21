import json
from typing import Dict, Optional, Any
from datetime import datetime

from base import BaseAgent
from configs import SimulationConfig

class FDAAgent(BaseAgent):
    """FDA regulatory agent (reactive mode)."""
    
    def __init__(self, config: SimulationConfig):
        super().__init__("fda_regulator", config)
        self.announcement_history = []
        self.mode = "reactive"  # Start with reactive mode
    
    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions for regulatory intervention."""
        
        system_prompt = """
        You are an FDA analyst monitoring drug shortage risks.
        Your role is to assess if public alerts are warranted to help resolve shortages.
        
        You operate in REACTIVE mode: respond to existing shortages, not predict future ones.
        """
        
        user_prompt = f"""
        Current Market Situation:
        - Period: {context.get('period', 0)}/{self.config.n_periods}
        - Total supply last period: {context.get('total_supply', 'Unknown')}
        - Total demand last period: {context.get('total_demand', 'Unknown')}
        - Shortage amount: {context.get('shortage_amount', 0)}
        - Shortage percentage: {context.get('shortage_percentage', 0):.1%}
        - Disrupted manufacturers: {len(context.get('disrupted_manufacturers', []))}/{self.config.n_manufacturers}
        
        For REACTIVE policy, focus on:
        - Confirmed shortages requiring coordination
        - Market failures needing intervention
        - Communication that encourages appropriate responses
        
        Analyze and respond with JSON:
        {{
            "role": "fda_regulator",
            "goal": "minimize_patient_impact_and_shortage_duration",
            "market_conditions": {{
                "shortage_status": "none/emerging/confirmed/critical",
                "supply_adequacy": "surplus/adequate/insufficient/critical",
                "market_stability": "stable/volatile/disrupted"
            }},
            "internal_state": {{
                "alert_urgency": "routine/elevated/high/critical",
                "intervention_threshold": "met/not_met",
                "communication_strategy": "none/watch/alert/urgent"
            }}
        }}
        """
        
        return await self.call_llm(
            system_prompt, user_prompt,
            expected_json_keys=["role", "market_conditions", "internal_state"]
        )
    
    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Make announcement decision."""
        
        system_prompt = """
        You are an FDA official responsible for drug shortage communications.
        Your announcements can significantly impact market behavior.
        
        Balance transparency with market stability. Consider whether announcements
        will help coordinate responses or potentially worsen shortages through panic.
        """
        
        user_prompt = f"""
        Market Analysis: {json.dumps(state_json, indent=2)}
        
        Decision Framework:
        1. No Action: Market conditions don't warrant intervention
        2. Monitoring Statement: Acknowledge awareness, encourage reporting
        3. Shortage Alert: Confirm shortage, provide stakeholder guidance
        4. Critical Shortage: Urgent communication, expedited support
        
        Consider:
        - Will announcement help resolve shortage faster?
        - Could announcement worsen situation via panic buying?
        - Are there coordination benefits from public information?
        - Patient safety risk of delayed communication?
        
        Respond with JSON:
        {{
            "decision": {{
                "announcement_type": "none/monitoring/alert/critical",
                "communication_urgency": "routine/elevated/urgent",
                "public_message": "brief announcement text if any"
            }},
            "reasoning": {{
                "shortage_assessment": "current status and trajectory",
                "intervention_justification": "why this response level",
                "market_impact_prediction": "expected stakeholder responses"
            }},
            "confidence": "low/moderate/high"
        }}
        """
        
        return await self.call_llm(
            system_prompt, user_prompt,
            expected_json_keys=["decision", "reasoning"]
        )
    
    def get_default_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative default: no announcement."""
        return {
            "decision": {
                "announcement_type": "none",
                "communication_urgency": "routine",
                "public_message": ""
            },
            "reasoning": {
                "shortage_assessment": "Default no action due to LLM failure",
                "intervention_justification": "Conservative approach when uncertain",
                "market_impact_prediction": "Avoid potential market disruption"
            },
            "confidence": "low"
        }
    
    def make_announcement(self, decision: Dict[str, Any]) -> Optional[str]:
        """Generate and record FDA announcement."""
        announcement_type = decision.get("decision", {}).get("announcement_type", "none")
        message = decision.get("decision", {}).get("public_message", "")
        
        if announcement_type == "none":
            return None
        
        # Record announcement
        self.announcement_history.append({
            "type": announcement_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        return message