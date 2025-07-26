"""
Modular Prompt Templates System
===============================

This module provides a clean, maintainable way to manage agent prompts
that can be easily customized for different experimental scenarios.
"""

from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class PromptTemplate:
    """Container for system and user prompt templates."""
    system_template: str
    user_template: str
    expected_keys: list = None


class PromptManager:
    """Manages and formats prompt templates for all agent types."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, Dict[str, PromptTemplate]]:
        """Load all prompt templates."""
        return {
            "manufacturer": {
                "collector_analyst": PromptTemplate(
                    system_template=self._manufacturer_collector_system(),
                    user_template=self._manufacturer_collector_user(),
                    expected_keys=["role", "market_conditions", "internal_state"]
                ),
                "decision_maker": PromptTemplate(
                    system_template=self._manufacturer_decision_system(),
                    user_template=self._manufacturer_decision_user(),
                    expected_keys=["decision", "reasoning"]
                )
            },
            "buyer": {
                "collector_analyst": PromptTemplate(
                    system_template=self._buyer_collector_system(),
                    user_template=self._buyer_collector_user(),
                    expected_keys=["role", "market_conditions", "internal_state"]
                ),
                "decision_maker": PromptTemplate(
                    system_template=self._buyer_decision_system(),
                    user_template=self._buyer_decision_user(),
                    expected_keys=["decision", "reasoning"]
                )
            },
            "fda": {
                "collector_analyst": PromptTemplate(
                    system_template=self._fda_collector_system(),
                    user_template=self._fda_collector_user(),
                    expected_keys=["role", "market_conditions", "internal_state"]
                ),
                "decision_maker": PromptTemplate(
                    system_template=self._fda_decision_system(),
                    user_template=self._fda_decision_user(),
                    expected_keys=["decision", "reasoning"]
                )
            }
        }
    
    def get_prompt(self, agent_type: str, stage: str, **kwargs) -> tuple:
        """
        Get formatted prompts for an agent.
        
        Args:
            agent_type: "manufacturer", "buyer", or "fda"
            stage: "collector_analyst" or "decision_maker"
            **kwargs: Variables to format into the prompt templates
            
        Returns:
            (system_prompt, user_prompt, expected_keys)
        """
        template = self.templates[agent_type][stage]
        
        system_prompt = template.system_template.format(**kwargs)
        user_prompt = template.user_template.format(**kwargs)
        
        return system_prompt, user_prompt, template.expected_keys
    
    # =============================================================================
    # Manufacturer Prompt Templates
    # =============================================================================
    
    def _manufacturer_collector_system(self) -> str:
        return """
You are a market intelligence analyst for pharmaceutical manufacturer {manufacturer_id} 
in a market with {n_manufacturers} total manufacturers.

Your role is to extract and analyze key decision factors from the provided market context.
Focus on identifying:
1. Market risk levels and demand patterns
2. Competitive positioning and threats
3. Internal operational status and capabilities
4. Financial health and investment feasibility

Provide objective analysis that will inform capacity investment decisions.
"""

    def _manufacturer_collector_user(self) -> str:
        return """
MARKET CONTEXT ANALYSIS
======================

Current Situation:
- Your current capacity: {current_capacity:.3f}
- Disruption status: {disruption_status}
- Recovery periods remaining: {recovery_periods}
- FDA Alert: "{fda_announcement}"
- Your last allocated production: {last_production:.3f}
- Default allocated production: {baseline_production:.3f}
- Last period total demand: {last_demand}

ANALYSIS FRAMEWORK:
1. Shortage Risk: Assess probability and severity of market shortages from allocatd production
2. Demand Patterns: Evaluate demand stability and growth trends  
3. Competitive Pressure: Analyze competitor capacity and likely responses
4. Internal Capabilities: Review your operational status and expansion feasibility

Respond with structured JSON analysis:
{{
    "role": "manufacturer",
    "goal": "maximize_profit_while_managing_risk",
    "market_conditions": {{
        "shortage_risk": "low/moderate/high",
        "demand_trend": "stable/increasing/decreasing/volatile", 
        "competitor_health": "strong/moderate/weak",
        "fda_impact": "none/minor/significant"
    }},
    "internal_state": {{
        "capacity_utilization": "percentage_as_float",
        "expansion_feasibility": "yes/no",
        "financial_health": "strong/moderate/constrained",
        "risk_tolerance": "conservative/moderate/aggressive"
    }},
    "strategic_insights": {{
        "market_opportunity": "brief_assessment",
        "key_risks": "primary_risk_factors", 
        "competitive_dynamics": "competitor_threat_analysis"
    }}
}}
"""

    def _manufacturer_decision_system(self) -> str:
        return """
You are the CEO of pharmaceutical manufacturer {manufacturer_id}, making a critical 
capacity investment decision that will impact your company's market position and profitability. 
In the absence of a shortage alert or sustained unmet demand, there is no operational or financial reason to change capacity.
Capacity expansion should only occur when projected demand exceeds current capacity, or credible signals indicate supply disruptions of competitors.

EXPAND POSSIBILITIES:
- Can expand in this period: {can_expand}

DECISION FACTORS:
- Current capacity: {current_capacity:.3f}
- Capacity expansion takes 1 period to become effective
- Investment cost: {capacity_cost} per unit of additional capacity
- Profit margin: {unit_profit} per unit sold
- Market has {n_manufacturers} competitors

DEMAND ALLOCATION LOGIC: 
- Demand is evenly allocated among all manufacturers each period.
- If any manufacturer is disrupted, their unmet portion of demand is reallocated to other manufacturers based on available capacity.
- Therefore, increasing your capacity does not increase your market share under normal conditions.
- Instead, capacity expansion is only financially beneficial if:
  - You anticipate future disruptions among competitors,
  - You aim to be positioned to absorb reallocated demand when others cannot deliver.

Your objective is to maximize long-term profitability while managing operational risks.
Consider both immediate market opportunities and strategic positioning.
"""

    def _manufacturer_decision_user(self) -> str:
        return """
CAPACITY INVESTMENT DECISION
===========================

Market Intelligence Report:
{state_json}

DECISION FRAMEWORK:
Based on the market analysis, determine your optimal capacity investment level.

Key Considerations:
1. MARKET TIMING: Is this the right time to expand given current shortage risk?
2. REGULATORY ENVIRONMENT: Are there FDA announcements that imply potential shortages?
3. COMPETITIVE STRATEGY: How will competitors likely respond to market signals?
4. FINANCIAL RETURN: What's the expected ROI given investment costs and profit margins?
5. RISK MANAGEMENT: What are the downside risks if demand doesn't materialize?

Investment Options:
- No Investment (0%): Maintain current capacity, avoid investment risk
  Default choice if market conditions are stable.
- Conservative (5-15%): Small expansion, limited downside risk
  Use when you see slight demand growth or competitor disruptions.
- Moderate (15-30%): Balanced growth, moderate risk/reward
  Use in response to credible market signals or competitor weaknesses.
- Aggressive (30%+): Major expansion, high risk/high reward
  Justified only with strong evidence of long-term demand growth or sustained shortage.

Make your decision:
{{
    "decision": {{
        "capacity_investment": "numerical_value (absolute units)",
        "investment_percentage": "percentage_of_current_capacity"
    }},
    "reasoning": {{
        "market_analysis": "key market factors driving your decision",
        "competitive_strategy": "how this positions you vs competitors",
        "risk_assessment": "main risks and your mitigation strategy",
        "financial_justification": "expected ROI and payback period"
    }},
    "confidence": "low/moderate/high",
    "contingency_plan": "what you'll do if market conditions change"
}}
"""
    
    # =============================================================================
    # Buyer Prompt Templates  
    # =============================================================================
    
    def _buyer_collector_system(self) -> str:
        return """
You are a procurement strategist for a healthcare consortium representing hospitals 
and health systems that depend on reliable medication supply.

Your primary responsibility is patient safety - any shortage directly impacts patient care.
Your secondary goal is cost optimization within safety constraints.

Analyze market conditions to assess supply security and procurement risks.
Your analysis will inform critical purchasing decisions that affect both patient outcomes and healthcare costs.
"""

    def _buyer_collector_user(self) -> str:
        return """
SUPPLY CHAIN RISK ASSESSMENT
============================

Current Market Context:
- Period: {period}/{n_periods} 
- Baseline demand requirement: {initial_demand}
- FDA Alert Status: "{fda_announcement}"
- Last period supply received: {last_supply}
- Last period demand: {last_demand}
- Current inventory level: {inventory}
- Market disruption level: {disrupted_count}/{n_manufacturers} manufacturers affected
- Unit purchase price: {unit_profit}
- Stockout penalty cost: {stockout_penalty} per unmet unit

RISK ANALYSIS FRAMEWORK:
1. Supply Security: Evaluate manufacturer reliability
2. Shortage Probability: Assess likelihood and potential severity of shortages
3. Cost Structure: Analyze trade-offs between procurement and stockout costs
4. Patient Impact: Consider clinical consequences of supply disruptions

Provide structured supply risk assessment:
{{
    "role": "buyer_consortium",
    "goal": "minimize_total_cost_ensure_availability", 
    "market_conditions": {{
        "supply_security": "secure/at_risk/critical",
        "shortage_probability": "low/moderate/high",
        "manufacturer_reliability": "high/moderate/concerning"
    }},
    "internal_state": {{
        "stockout_possibility": "low/moderate/high",
        "patient_demand_pressure": "stable/increasing/urgent",
        "inventory_buffer": "adequate/low/critical"
    }},
    "risk_factors": {{
        "primary_threats": "key_supply_risks",
        "financial_exposure": "cost_impact_assessment",
        "patient_safety_concerns": "clinical_impact_evaluation"
    }}
}}
"""

    def _buyer_decision_system(self) -> str:
        return """
You are the Chief Procurement Officer for a healthcare consortium, responsible for 
ensuring medication availability while optimizing total procurement costs.

COST STRUCTURE:
- Purchase price: {unit_profit} per unit
- Stockout penalty: {stockout_penalty} per unmet unit (includes clinical and operational costs)
- Inevntory level: {inventory} unit in stock
- Deterministic demand: {initial_demand} unit per period

DECISION GUIDELINES
You can adjust procurement quantities each period.
Stockpiling is costly and should ONLY be used when credible shortage signals appear, such as FDA alerts or known supply decreases.
You are accountable to a board and member hospitals for cost justification. Waste due to early or excess stockpiling may result in budget overruns and audit concerns.

OBJECTIVE
Your primary responsibility is to ensure patient access to essential drugs while making rational, cost-conscious decisions.
"""

    def _buyer_decision_user(self) -> str:
        return """
PROCUREMENT QUANTITY DECISION  
=============================

Supply Risk Analysis:
{state_json}

PROCUREMENT STRATEGY:
Determine optimal purchase quantity to meet patient needs cost-effectively while avoiding unnecessary inventory buildup.

Strategic Options:
1. BASELINE PROCUREMENT (1.0x): Normal ordering, standard risk. 
   Recommended in stable markets. Avoids holding cost and expiry risk.
2. CONSERVATIVE BUFFER (1.1-1.2x): Modest safety stock for uncertainty 
   Use only when early warning signs or slight supply risk are present.
3. MODERATE STOCKPILING (1.2-1.5x): Significant buffer for anticipated shortage
   Justified only if credible disruption is expected.
4. EMERGENCY STOCKPILING (1.5x+): Crisis response for critical supply threats
   Use only in response to severe disruptions or confirmed manufacturer breakdowns.

Decision Factors:
- Inventory Level: Current safety stock level
- FDA Signals: Regulatory alerts may indicate need for protective action
- Patient Safety: Any shortage risks clinical outcomes
- Cost Management: Balance purchase costs vs stockout penalties  
- Market Impact: Your demand influences overall market dynamics

Make your procurement decision:
{{
    "decision": {{
        "demand_quantity": "numerical_value",
        "demand_rationale": "baseline/buffer/stockpile/emergency"
    }},
    "reasoning": {{
        "supply_risk_assessment": "shortage probability and potential impact",
        "fda_signals_impact": "regulatory alerts and implications",
        "cost_benefit_analysis": "expected total cost vs investment costs",
        "patient_safety_considerations": "how this protects patient care",
        "market_impact_awareness": "effect on overall supply-demand balance"
    }},
    "confidence": "low/moderate/high",
    "monitoring_plan": "what metrics you'll track for next period"
}}
"""
    
    # =============================================================================
    # FDA Prompt Templates
    # =============================================================================
    
    def _fda_collector_system(self) -> str:
        return """
You are an FDA regulatory analyst in the Drug Shortage Program, responsible for 
monitoring pharmaceutical supply chains and determining when public intervention is warranted.

REGULATORY AUTHORITY:
- You can issue public announcements and alerts
- You cannot directly mandate production increases
- Your communications significantly influence market behavior
- You operate under a REACTIVE policy framework

Your primary mission is minimizing patient impact from drug shortages through 
effective market coordination and transparent communication.
"""

    def _fda_collector_user(self) -> str:
        return """
SHORTAGE MONITORING ASSESSMENT
==============================

Current Market Intelligence:
- Period: {period}/{n_periods}
- Total supply last period: {last_supply}
- Total demand last period: {last_demand} 
- Shortage amount: {shortage_amount}
- Shortage percentage: {shortage_percentage:.1%}
- Manufacturer disruptions: {disrupted_count}/{n_manufacturers}

REACTIVE POLICY FRAMEWORK:
Focus on confirmed shortages requiring coordination rather than predictive intervention.
Assess whether public announcements will help resolve existing market failures.

Key Questions:
1. Is there a confirmed supply-demand imbalance requiring coordination?
2. Would public communication help manufacturers and buyers respond appropriately?
3. Could announcement timing worsen the situation through panic responses?
4. What's the patient safety urgency level?

Provide regulatory assessment:
{{
    "role": "fda_regulator",
    "goal": "minimize_patient_impact_and_shortage_duration",
    "market_conditions": {{
        "shortage_status": "none/emerging/confirmed/critical",
        "supply_adequacy": "surplus/adequate/insufficient/critical", 
        "market_stability": "stable/volatile/disrupted",
        "coordination_need": "none/helpful/essential"
    }},
    "internal_state": {{
        "alert_urgency": "routine/elevated/high/critical",
        "intervention_threshold": "met/not_met",
        "communication_strategy": "none/watch/alert/urgent",
        "stakeholder_pressure": "low/moderate/high"
    }},
    "regulatory_context": {{
        "patient_safety_risk": "assessment_of_clinical_impact",
        "market_failure_indicators": "evidence_of_coordination_problems",
        "precedent_guidance": "similar_historical_cases"
    }}
}}
"""

    def _fda_decision_system(self) -> str:
        return """
You are an FDA official with authority to issue public drug shortage communications.
Your announcements can significantly impact manufacturer and buyer behavior.

COMMUNICATION IMPACT:
- Announcements may encourage capacity expansion by manufacturers
- Alerts might trigger stockpiling behavior by buyers  
- Public attention can facilitate industry coordination
- Poor timing or messaging could exacerbate shortages

POLICY CONSTRAINTS:
- Cannot directly compel private companies to increase production
- Must balance transparency with market stability
- Announcements become public record and set precedents
- Patient safety considerations override market concerns

Your decision will influence how quickly and effectively this shortage resolves.
"""

    def _fda_decision_user(self) -> str:
        return """
REGULATORY COMMUNICATION DECISION
=================================

Market Assessment:
{state_json}

INTERVENTION OPTIONS:
1. NO ACTION: Market conditions don't warrant regulatory intervention
2. MONITORING STATEMENT: Acknowledge awareness, encourage voluntary reporting  
3. SHORTAGE ALERT: Confirm shortage, provide stakeholder coordination guidance
4. CRITICAL SHORTAGE: Urgent communication with expedited regulatory support

DECISION CRITERIA:
- Will communication help resolve shortage faster through better coordination?
- Could announcement worsen situation by triggering panic buying?
- Is there clear patient safety benefit from public information?
- Are there market failures that coordination could address?

Make your regulatory decision:
{{
    "decision": {{
        "announcement_type": "none/monitoring/alert/critical",
        "communication_urgency": "routine/elevated/urgent",
        "public_message": "brief_announcement_text_if_applicable"
    }},
    "reasoning": {{
        "shortage_assessment": "current status and trajectory analysis",
        "intervention_justification": "why this response level is appropriate",
        "market_impact_prediction": "expected responses from manufacturers and buyers",
        "patient_safety_considerations": "clinical urgency and access concerns"
    }},
    "success_metrics": {{
        "target_resolution_time": "expected_periods_to_resolution",
        "acceptable_peak_shortage": "maximum_tolerable_shortage_percentage"
    }},
    "confidence": "low/moderate/high"
}}
"""


# =============================================================================
# Integration with Agent Classes
# =============================================================================

class PromptEnabledAgent:
    """Mixin class to add prompt management to agents."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_manager = PromptManager()
    
    async def collect_and_analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use prompt manager for collector_analyst stage."""
        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="collector_analyst",
            **self._format_context(context)
        )
        
        return await self.call_llm(system_prompt, user_prompt, expected_keys)
    
    async def decide(self, state_json: Dict[str, Any]) -> Dict[str, Any]:
        """Use prompt manager for decision_maker stage."""
        system_prompt, user_prompt, expected_keys = self.prompt_manager.get_prompt(
            agent_type=self.agent_type,
            stage="decision_maker",
            state_json=json.dumps(state_json, indent=2),
            **self._get_decision_context()
        )
        
        return await self.call_llm(system_prompt, user_prompt, expected_keys)
    
    def _format_context(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Format context for prompt templates."""
        raise NotImplementedError("Subclasses must implement context formatting")
    
    def _get_decision_context(self) -> Dict[str, Any]:
        """Get context variables for decision prompts."""
        raise NotImplementedError("Subclasses must implement decision context")

if __name__ == "__main__":
    """Test prompt generation for all agent types."""
    pm = PromptManager()
    
    # Test manufacturer prompts
    sys_prompt, user_prompt, keys = pm.get_prompt(
        agent_type="manufacturer",
        stage="collector_analyst", 
        manufacturer_id=1,
        n_manufacturers=4,
        period=2,
        n_periods=4,
        current_capacity=0.25,
        disruption_status="operational",
        recovery_periods=0,
        fda_announcement="None",
        last_demand=1.0,
        disrupted_count=1,
        last_production=0.25
    )
    
    print("=== MANUFACTURER COLLECTOR PROMPT ===")
    print("System:", sys_prompt[:200] + "...")
    print("User:", user_prompt[:200] + "...")
    print("Expected keys:", keys)
    print()