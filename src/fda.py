import json
from typing import Dict, Optional, Any
from datetime import datetime

from base import BaseAgent
from prompts import PromptManager
from configs import SimulationConfig

class FDAAgent(BaseAgent):
    """FDA regulatory agent (reactive mode)."""

    def __init__(self, config: SimulationConfig):
        super().__init__("fda_regulator", config)
        self.agent_type = "fda"  # For prompt manager
        self.prompt_manager = PromptManager()
        
        self.announcement_history = []
        self.mode = "reactive"

    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for regulatory analysis."""

        prompt_context = {
            "period": context.get('period', 0),
            "n_periods": self.config.n_periods,
            "last_supply": context.get('last_supply', 'Unknown'),
            "last_demand": context.get('last_demand', 'Unknown'),
            "shortage_amount": context.get('shortage_amount', 0),
            "shortage_percentage": context.get('shortage_percentage', 0),
            "disrupted_count": len(context.get('disrupted_manufacturers', [])),
            "n_manufacturers": self.config.n_manufacturers
        }

        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="collector_analyst",
            **prompt_context
        )

        return await self.call_llm(system_prompt, user_prompt, expected_keys)

    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for announcement decisions."""

        decision_context = {
            "state_json": json.dumps(state_json, indent=2)
        }

        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="decision_maker",
            **decision_context
        )

        return await self.call_llm(system_prompt, user_prompt, expected_keys)

    # Keep existing methods unchanged
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
