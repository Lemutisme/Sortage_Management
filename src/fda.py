import json
from typing import Dict, Optional, Any
from datetime import datetime

from base import BaseAgent
from prompts import PromptManager
from configs import SimulationConfig

class FDAAgent(BaseAgent):
    """FDA regulatory agent (reactive mode) with comprehensive logging."""

    def __init__(self, config: SimulationConfig):
        super().__init__("fda_regulator", config)
        self.agent_type = "fda"  # For prompt manager
        self.prompt_manager = PromptManager()
        
        self.announcement_history = []
        self.mode = "reactive"

    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for regulatory analysis with detailed logging."""
        
        self.logger.info("FDA starting regulatory market assessment")
        
        # Extract key market indicators for logging
        shortage_amount = context.get('shortage_amount', 0)
        shortage_percentage = context.get('shortage_percentage', 0)
        disrupted_count = len(context.get('disrupted_manufacturers', []))
        
        self.logger.debug(
            f"Market indicators - Shortage: {shortage_amount:.3f} "
            f"({shortage_percentage:.1%}), "
            f"Disrupted manufacturers: {disrupted_count}"
        )

        prompt_context = {
            "period": context.get('period', 0),
            "n_periods": self.config.n_periods,
            "last_supply": context.get('last_supply', 'Unknown'),
            "last_demand": context.get('last_demand', 'Unknown'),
            "shortage_amount": shortage_amount,
            "shortage_percentage": shortage_percentage,
            "disrupted_count": disrupted_count,
            "n_manufacturers": self.config.n_manufacturers
        }

        self.logger.debug(f"Regulatory analysis context: {prompt_context}")

        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="collector_analyst",
            **prompt_context
        )

        self.logger.debug("Calling LLM for regulatory analysis")
        
        result = await self.call_llm(
            system_prompt, 
            user_prompt, 
            expected_keys,
            stage="regulatory_analysis"
        )
        
        # Log key regulatory assessments
        market_conditions = result.get('market_conditions', {})
        shortage_status = market_conditions.get('shortage_status', 'unknown')
        supply_adequacy = market_conditions.get('supply_adequacy', 'unknown')
        
        internal_state = result.get('internal_state', {})
        alert_urgency = internal_state.get('alert_urgency', 'unknown')
        intervention_threshold = internal_state.get('intervention_threshold', 'unknown')
        
        self.logger.info(
            f"Regulatory analysis completed - "
            f"Shortage status: {shortage_status}, "
            f"Supply adequacy: {supply_adequacy}, "
            f"Alert urgency: {alert_urgency}, "
            f"Intervention threshold: {intervention_threshold}"
        )
        
        return result

    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Use modular prompt system for announcement decisions with detailed logging."""
        
        self.logger.info("FDA making regulatory communication decision")
        
        # Log decision context
        self.logger.debug(f"Previous announcements: {len(self.announcement_history)}")
        if self.announcement_history:
            last_announcement = self.announcement_history[-1]
            self.logger.debug(f"Last announcement: {last_announcement}")

        decision_context = {
            "state_json": json.dumps(state_json, indent=2)
        }

        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="decision_maker",
            **decision_context
        )

        self.logger.debug("Calling LLM for announcement decision")
        
        result = await self.call_llm(
            system_prompt, 
            user_prompt, 
            expected_keys,
            stage="announcement_decision"
        )
        
        # Log decision outcome
        decision = result.get("decision", {})
        announcement_type = decision.get("announcement_type", "none")
        communication_urgency = decision.get("communication_urgency", "routine")
        public_message = decision.get("public_message", "")
        confidence = result.get("confidence", "unknown")
        
        self.logger.info(
            f"Announcement decision made - "
            f"Type: {announcement_type}, "
            f"Urgency: {communication_urgency}, "
            f"Confidence: {confidence}"
        )
        
        if announcement_type != "none":
            self.logger.info(f"Public message prepared: '{public_message[:100]}...'")
        else:
            self.logger.info("No public announcement will be made")
        
        # Log reasoning if available
        reasoning = result.get("reasoning", {})
        if reasoning:
            self.logger.debug(f"Decision reasoning: {json.dumps(reasoning, indent=2)}")
            
            # Log specific reasoning components
            shortage_assessment = reasoning.get("shortage_assessment", "")
            intervention_justification = reasoning.get("intervention_justification", "")
            
            if shortage_assessment:
                self.logger.debug(f"Shortage assessment: {shortage_assessment}")
            if intervention_justification:
                self.logger.debug(f"Intervention justification: {intervention_justification}")
        
        return result

    def get_default_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative default: no announcement with logging."""
        self.logger.warning("Using default FDA decision due to LLM failure")
        
        return {
            "decision": {
                "announcement_type": "none",
                "communication_urgency": "routine",
                "public_message": ""
            },
            "reasoning": {
                "shortage_assessment": "Default no action due to LLM failure - insufficient information",
                "intervention_justification": "Conservative approach when uncertain to avoid market disruption",
                "market_impact_prediction": "Avoid potential panic responses from stakeholders",
                "patient_safety_considerations": "Monitor situation without intervention until more data available"
            },
            "success_metrics": {
                "target_resolution_time": "unknown",
                "acceptable_peak_shortage": "unknown"
            },
            "confidence": "low"
        }

    def make_announcement(self, decision: Dict[str, Any]) -> Optional[str]:
        """Generate and record FDA announcement with comprehensive logging."""
        
        announcement_type = decision.get("decision", {}).get("announcement_type", "none")
        message = decision.get("decision", {}).get("public_message", "")
        communication_urgency = decision.get("decision", {}).get("communication_urgency", "routine")

        if announcement_type == "none":
            self.logger.info("No FDA announcement issued")
            return None

        # Create announcement record
        announcement_record = {
            "type": announcement_type,
            "message": message,
            "urgency": communication_urgency,
            "timestamp": datetime.now().isoformat(),
            "period": self.current_period,
            "decision_context": decision.get("reasoning", {})
        }
        
        self.announcement_history.append(announcement_record)
        
        # Log announcement details
        self.logger.warning(
            f"FDA ANNOUNCEMENT ISSUED - "
            f"Type: {announcement_type.upper()}, "
            f"Urgency: {communication_urgency.upper()}"
        )
        
        self.logger.warning(f"Public message: {message}")
        
        # Log expected impact
        reasoning = decision.get("reasoning", {})
        market_impact = reasoning.get("market_impact_prediction", "")
        if market_impact:
            self.logger.info(f"Expected market impact: {market_impact}")
        
        # Log success metrics if available
        success_metrics = decision.get("success_metrics", {})
        if success_metrics:
            target_resolution = success_metrics.get("target_resolution_time", "")
            acceptable_shortage = success_metrics.get("acceptable_peak_shortage", "")
            
            if target_resolution:
                self.logger.info(f"Target resolution time: {target_resolution}")
            if acceptable_shortage:
                self.logger.info(f"Acceptable peak shortage: {acceptable_shortage}")

        return message

    def get_announcement_history_summary(self) -> Dict[str, Any]:
        """Get announcement history summary for analysis."""
        
        if not self.announcement_history:
            return {
                "total_announcements": 0,
                "announcement_types": {},
                "periods_with_announcements": []
            }
        
        # Count announcement types
        type_counts = {}
        urgency_counts = {}
        periods_with_announcements = []
        
        for announcement in self.announcement_history:
            announcement_type = announcement.get("type", "unknown")
            urgency = announcement.get("urgency", "unknown")
            period = announcement.get("period", -1)
            
            type_counts[announcement_type] = type_counts.get(announcement_type, 0) + 1
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            
            if period >= 0:
                periods_with_announcements.append(period)
        
        summary = {
            "total_announcements": len(self.announcement_history),
            "announcement_types": type_counts,
            "urgency_distribution": urgency_counts,
            "periods_with_announcements": periods_with_announcements,
            "first_announcement_period": min(periods_with_announcements) if periods_with_announcements else None,
            "last_announcement_period": max(periods_with_announcements) if periods_with_announcements else None
        }
        
        self.logger.debug(f"Announcement history summary: {json.dumps(summary, indent=2)}")
        return summary

    def evaluate_intervention_effectiveness(self, market_trajectory: list) -> Dict[str, Any]:
        """Evaluate the effectiveness of FDA interventions."""
        
        if not self.announcement_history or not market_trajectory:
            return {"status": "insufficient_data"}
        
        announcement_periods = [a.get("period", -1) for a in self.announcement_history]
        announcement_periods = [p for p in announcement_periods if p >= 0]
        
        if not announcement_periods:
            return {"status": "no_valid_announcement_periods"}
        
        # Analyze shortage trends before and after announcements
        effectiveness_analysis = {
            "announcement_periods": announcement_periods,
            "total_announcements": len(self.announcement_history),
            "market_periods": len(market_trajectory)
        }
        
        # For each announcement, look at shortage trends
        for i, period in enumerate(announcement_periods):
            if period < len(market_trajectory):
                pre_announcement_shortage = 0
                post_announcement_shortage = 0
                
                # Look at shortage before announcement
                if period > 0:
                    pre_announcement_shortage = market_trajectory[period - 1].get("shortage_percentage", 0)
                
                # Look at shortage after announcement (if available)
                if period + 1 < len(market_trajectory):
                    post_announcement_shortage = market_trajectory[period + 1].get("shortage_percentage", 0)
                
                effectiveness_analysis[f"announcement_{i+1}"] = {
                    "period": period,
                    "type": self.announcement_history[i].get("type", "unknown"),
                    "pre_shortage": pre_announcement_shortage,
                    "post_shortage": post_announcement_shortage,
                    "shortage_change": post_announcement_shortage - pre_announcement_shortage
                }
        
        self.logger.info(f"Intervention effectiveness analysis completed")
        return effectiveness_analysis