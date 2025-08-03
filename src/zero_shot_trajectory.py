import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import openai
from datetime import datetime

from configs import SimulationConfig
from base import BaseAgent


@dataclass
class TrajectoryPeriod:
    """Data for a single period in the trajectory."""
    period: int
    total_demand: float
    total_supply: float
    shortage_amount: float
    unsold: float
    shortage_percentage: float
    disrupted_manufacturers: List[int]
    fda_announcement: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SupplyTrajectory:
    """Complete supply trajectory prediction."""
    periods: int
    scenario_summary: Dict[str, Any]
    trajectory: List[TrajectoryPeriod]
    predicted_resolution_period: int
    confidence_level: str
    economic_reasoning: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "periods": self.periods,
            "scenario_summary": self.scenario_summary,
            "trajectory": [period.to_dict() for period in self.trajectory],
            "predicted_resolution_period": self.predicted_resolution_period,
            "confidence_level": self.confidence_level,
            "economic_reasoning": self.economic_reasoning
        }


@dataclass
class TrajectoryScenarioContext:
    """Enhanced scenario context for trajectory prediction."""
    n_manufacturers: int
    periods: int
    disruption_prob: float
    disruption_magnitude: float
    disrupted_manufacturers: List[int]
    initial_demand: float
    market_structure: str
    
    # Cost structure parameters
    capacity_cost: float
    unit_profit: float
    holding_cost: float
    stockout_penalty: float
    
    # Initial conditions
    initial_supply: float
    initial_shortage_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ZeroShotTrajectoryPredictor:
    """Zero-shot baseline for predicting complete supply trajectories using LLM."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.logger = logging.getLogger("ZeroShotTrajectoryPredictor")
        
        # Setup OpenAI client
        if config.api_key:
            openai.api_key = config.api_key
            if config.azure_endpoint:
                openai.api_base = config.azure_endpoint
                openai.api_type = "gpt-4o"
                openai.api_version = config.api_version
        else:
            self.logger.warning("No API key found - will use mock responses")
    
    def create_scenario_context(self, 
                              n_manufacturers: int,
                              periods: int,
                              disruption_prob: float,
                              disruption_magnitude: float,
                              disrupted_manufacturers: List[int],
                              initial_demand: float = 1.0,
                              initial_supply: float = None,
                              initial_shortage_pct: float = 0.0) -> TrajectoryScenarioContext:
        """Create scenario context for trajectory prediction."""
        
        market_structure = "concentrated" if n_manufacturers <= 3 else "competitive"
        
        # Estimate initial supply if not provided
        if initial_supply is None:
            if disrupted_manufacturers:
                # Account for disrupted capacity
                disrupted_capacity_loss = len(disrupted_manufacturers) * disruption_magnitude / n_manufacturers
                initial_supply = initial_demand * (1 - disrupted_capacity_loss)
            else:
                initial_supply = initial_demand
        
        return TrajectoryScenarioContext(
            n_manufacturers=n_manufacturers,
            periods=periods,
            disruption_prob=disruption_prob,
            disruption_magnitude=disruption_magnitude,
            disrupted_manufacturers=disrupted_manufacturers or [],
            initial_demand=initial_demand,
            market_structure=market_structure,
            capacity_cost=self.config.capacity_cost,
            unit_profit=self.config.unit_profit,
            holding_cost=self.config.holding_cost,
            stockout_penalty=self.config.stockout_penalty,
            initial_supply=initial_supply,
            initial_shortage_percentage=initial_shortage_pct
        )
    
    def _create_trajectory_prompt(self, scenario: TrajectoryScenarioContext) -> Tuple[str, str]:
        """Create system and user prompts for trajectory prediction."""
        
        system_prompt = """You are an expert pharmaceutical supply chain economist and forecasting specialist. Your task is to predict the complete supply and demand trajectory over multiple periods during a drug shortage scenario.

You have deep expertise in:
- Dynamic supply chain modeling and capacity expansion decisions
- Pharmaceutical manufacturing economics and investment timing
- Healthcare buyer behavior and inventory management during shortages
- FDA regulatory interventions and their market timing effects
- Game theory and competitive dynamics in oligopolistic pharmaceutical markets

Provide period-by-period predictions with clear economic reasoning for each transition."""

        user_prompt = f"""
Predict the complete supply-demand trajectory for the following drug shortage scenario.

## Scenario Parameters
- **Market Structure**: {scenario.n_manufacturers} manufacturers
- **Simulation Periods**: {scenario.periods} periods total
- **Initial Demand**: {scenario.initial_demand:.2f} units
- **Initial Supply**: {scenario.initial_supply:.2f} units
- **Initial Shortage**: {scenario.initial_shortage_percentage:.1%}

## Disruption Context
- **Disruption Probability**: {scenario.disruption_prob:.2%} per period
- **Disruption Magnitude**: {scenario.disruption_magnitude:.1%} capacity loss
- **Currently Disrupted**: {scenario.disrupted_manufacturers} (manufacturers by ID)

## Economic Parameters
- **Capacity Investment Cost**: ${scenario.capacity_cost} per unit of capacity
- **Unit Profit Margin**: ${scenario.unit_profit}
- **Inventory Holding Cost**: ${scenario.holding_cost} per unit per period
- **Stockout Penalty**: ${scenario.stockout_penalty} per unit shortage

## Information Privacy
- manufacturers do not know competitors' investments or disruptions until they are announced
- FDA does not know manufacturers' capacity and investment plans
- FDA announcements are made periodically to inform manufacturers and buyers of the situation

## Investment and Recovery
- Manufacturers can invest in capacity each period, but investments take 1 period to become effective.
- Disrupted manufacturers recover after their disruption period ends, returning to full capacity.
- Competition may avoid investments if they expect competitors to do so, leading to strategic delays.

## Prediction Task

Predict the period-by-period evolution of:
1. **Total Demand** - How demand evolves (panic buying, stockpiling, normalization)
2. **Total Supply** - How supply recovers (capacity investments, disruption recovery)
3. **Shortage Amount** - Unmet demand each period
4. **Shortage Percentage** - (Shortage Amount / Total Demand) * 100
5. **Disrupted Manufacturers** - Which manufacturers remain disrupted
6. **FDA Announcements** - Regulatory communications each period

## Economic Reasoning Framework

Consider these dynamic factors:

**Supply Evolution:**
- **Capacity Investment Timing**: When do manufacturers invest based on shortage profits?
- **Disruption Recovery**: How long do disruptions persist?
- **Competitive Response**: How does market structure affect investment coordination?
- **Regulatory Pressure**: How do FDA announcements accelerate manufacturer response?

**Demand Evolution:**
- **Panic Buying**: Initial demand spikes due to shortage awareness
- **Inventory Building**: Buyers increase safety stock during uncertainty
- **Demand Normalization**: Return to baseline as shortage resolves

**Market Dynamics:**
- **Game Theory**: Strategic interactions between manufacturers
- **Information Cascades**: How shortage signals propagate
- **Regulatory Coordination**: FDA role in market coordination

## Output Format

Provide your prediction as a JSON object with this exact structure:

```json
{{
    "trajectory_prediction": [
        {{
            "period": 0,
            "total_demand": <float>,
            "total_supply": <float>,
            "shortage_amount": <float>,
            "unsold": <float>,
            "shortage_percentage": <float>,
            "disrupted_manufacturers": [<list of manufacturer IDs>],
            "fda_announcement": "<string or empty>",
            "reasoning": "<brief explanation of this period's dynamics>"
        }},
        {{
            "period": 1,
            "total_demand": <float>,
            "total_supply": <float>,
            "shortage_amount": <float>,
            "unsold": <float>,
            "shortage_percentage": <float>,
            "disrupted_manufacturers": [<list of manufacturer IDs>],
            "fda_announcement": "<string or empty>",
            "reasoning": "<brief explanation of this period's dynamics>"
        }},
        ... (continue for all {scenario.periods} periods)
    ],
    "predicted_resolution_period": <integer - when shortage_percentage reaches ~0>,
    "confidence_level": "<high|medium|low>",
    "economic_reasoning": {{
        "supply_recovery_mechanism": "<how supply will recover>",
        "demand_evolution_pattern": "<how demand will evolve>",
        "market_coordination": "<how manufacturers will coordinate>",
        "regulatory_impact": "<FDA intervention effects>",
        "key_turning_points": ["<period X: event>", "<period Y: event>"]
    }},
    "scenario_summary": {{
        "peak_shortage_period": <integer>,
        "peak_shortage_percentage": <float>,
        "total_shortage_periods": <integer>,
        "supply_recovery_periods": <integer>
    }}
}}
```

## Key Constraints

1. **Conservation**: shortage_amount = max(0, total_demand - total_supply)
2. **Percentage**: shortage_percentage = (shortage_amount / total_demand) * 100
3. **Unsold**: unsold = max(0, total_supply - total_demand)
4. **Realism**: Supply cannot increase instantaneously - capacity investments take time
5. **Consistency**: Disrupted manufacturers list should evolve logically over time

Focus on economic fundamentals and realistic market dynamics. Explain how profit incentives, competitive pressures, and regulatory signals drive the trajectory evolution.
"""

        return system_prompt, user_prompt
    
    async def predict_supply_trajectory(self, scenario: TrajectoryScenarioContext) -> SupplyTrajectory:
        """Predict complete supply trajectory using zero-shot LLM reasoning."""
        
        self.logger.info(f"Making trajectory prediction for {scenario.periods} periods with {scenario.n_manufacturers} manufacturers")
        
        system_prompt, user_prompt = self._create_trajectory_prompt(scenario)
        
        try:
            if not self.config.api_key:
                # Return mock response for testing
                return self._get_mock_trajectory(scenario)
            
            # Make LLM API call
            response = await self._call_llm(system_prompt, user_prompt)
            trajectory = self._parse_trajectory_response(response, scenario)
            
            self.logger.info(f"Predicted trajectory with resolution at period {trajectory.predicted_resolution_period}")
            
            return trajectory
            
        except Exception as e:
            self.logger.error(f"Trajectory prediction failed: {e}")
            return self._get_fallback_trajectory(scenario)
    
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make async call to LLM API."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            if self.config.azure_endpoint:
                response = await openai.ChatCompletion.acreate(
                    engine=self.config.llm_model,
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    max_tokens=4000  # Increased for trajectory data
                )
            else:
                response = await openai.ChatCompletion.acreate(
                    model=self.config.llm_model,
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    max_tokens=4000
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_trajectory_response(self, response: str, scenario: TrajectoryScenarioContext) -> SupplyTrajectory:
        """Parse LLM response into structured trajectory."""
        
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                prediction_data = json.loads(json_str)
                
                # Validate and extract trajectory
                trajectory_data = prediction_data.get('trajectory_prediction', [])
                
                if len(trajectory_data) != scenario.periods:
                    raise ValueError(f"Expected {scenario.periods} periods, got {len(trajectory_data)}")
                
                # Create trajectory periods
                trajectory_periods = []
                for period_data in trajectory_data:
                    period = TrajectoryPeriod(
                        period=period_data['period'],
                        total_demand=float(period_data['total_demand']),
                        total_supply=float(period_data['total_supply']),
                        shortage_amount=float(period_data['shortage_amount']),
                        unsold=float(period_data['unsold']),
                        shortage_percentage=float(period_data['shortage_percentage']),
                        disrupted_manufacturers=period_data.get('disrupted_manufacturers', []),
                        fda_announcement=period_data.get('fda_announcement', '')
                    )
                    trajectory_periods.append(period)
                
                # Create complete trajectory
                trajectory = SupplyTrajectory(
                    periods=scenario.periods,
                    scenario_summary=prediction_data.get('scenario_summary', {}),
                    trajectory=trajectory_periods,
                    predicted_resolution_period=prediction_data.get('predicted_resolution_period', scenario.periods),
                    confidence_level=prediction_data.get('confidence_level', 'medium'),
                    economic_reasoning=prediction_data.get('economic_reasoning', {})
                )
                
                return trajectory
                
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.warning(f"Failed to parse trajectory response: {e}")
            # Fall back to heuristic generation
            return self._get_fallback_trajectory(scenario)
    
    def _get_mock_trajectory(self, scenario: TrajectoryScenarioContext) -> SupplyTrajectory:
        """Generate mock trajectory for testing without API key."""
        
        self.logger.info("Generating mock trajectory using heuristic logic")
        
        trajectory_periods = []
        current_demand = scenario.initial_demand
        current_supply = scenario.initial_supply
        disrupted_manufacturers = scenario.disrupted_manufacturers.copy()
        
        for period in range(scenario.periods):
            # Demand evolution
            if period == 0:
                # Initial panic buying if shortage exists
                if scenario.initial_shortage_percentage > 0.1:
                    current_demand *= 1.2  # 20% panic buying
            elif period == 1:
                # Continued elevated demand
                current_demand *= 1.1
            else:
                # Gradual normalization
                current_demand *= 0.95
                current_demand = max(current_demand, scenario.initial_demand)
            
            # Supply evolution
            if period == 0:
                # Starting supply
                pass
            elif period >= 2:
                # Capacity investments start to take effect
                if scenario.market_structure == "competitive":
                    supply_increase = 0.15  # Faster response in competitive markets
                else:
                    supply_increase = 0.10  # Slower in concentrated markets
                
                current_supply += supply_increase
                
                # Disruption recovery
                if disrupted_manufacturers and period >= 3:
                    # Some manufacturers recover
                    if len(disrupted_manufacturers) > 1:
                        disrupted_manufacturers = disrupted_manufacturers[:-1]
                    else:
                        disrupted_manufacturers = []
            
            # Calculate shortage metrics
            shortage_amount = max(0, current_demand - current_supply)
            unsold = max(0, current_supply - current_demand)
            shortage_percentage = (shortage_amount / current_demand) * 100 if current_demand > 0 else 0
            
            # FDA announcements
            fda_announcement = ""
            if period == 0 and shortage_percentage > 10:
                fda_announcement = "The FDA is monitoring emerging supply disruptions and encourages manufacturer reporting."
            elif period == 1 and shortage_percentage > 15:
                fda_announcement = "The FDA is actively coordinating with manufacturers to address the ongoing shortage."
            elif period >= 2 and shortage_percentage > 20:
                fda_announcement = "The FDA urges increased production and coordination among stakeholders."
            
            period_data = TrajectoryPeriod(
                period=period,
                total_demand=round(current_demand, 3),
                total_supply=round(current_supply, 3),
                shortage_amount=round(shortage_amount, 3),
                unsold=round(unsold, 3),
                shortage_percentage=round(shortage_percentage, 1),
                disrupted_manufacturers=disrupted_manufacturers.copy(),
                fda_announcement=fda_announcement
            )
            
            trajectory_periods.append(period_data)
        
        # Determine resolution period
        resolution_period = scenario.periods
        for i, period_data in enumerate(trajectory_periods):
            if period_data.shortage_percentage < 5.0:  # Consider <5% as resolved
                resolution_period = i
                break
        
        return SupplyTrajectory(
            periods=scenario.periods,
            scenario_summary={
                "peak_shortage_period": max(range(len(trajectory_periods)), 
                                          key=lambda i: trajectory_periods[i].shortage_percentage),
                "peak_shortage_percentage": max(p.shortage_percentage for p in trajectory_periods),
                "total_shortage_periods": sum(1 for p in trajectory_periods if p.shortage_percentage > 0),
                "supply_recovery_periods": resolution_period
            },
            trajectory=trajectory_periods,
            predicted_resolution_period=resolution_period,
            confidence_level="medium",
            economic_reasoning={
                "supply_recovery_mechanism": "Gradual capacity expansion driven by shortage profits",
                "demand_evolution_pattern": "Initial panic buying followed by normalization",
                "market_coordination": f"{scenario.market_structure} market affects coordination speed",
                "regulatory_impact": "FDA monitoring encourages proactive manufacturer response",
                "mock_response": True
            }
        )
    
    def _get_fallback_trajectory(self, scenario: TrajectoryScenarioContext) -> SupplyTrajectory:
        """Generate simple fallback trajectory when LLM fails."""
        
        trajectory_periods = []
        for period in range(scenario.periods):
            # Very simple linear recovery
            recovery_rate = period / max(scenario.periods - 1, 1)
            current_supply = scenario.initial_supply + (scenario.initial_demand - scenario.initial_supply) * recovery_rate
            current_demand = scenario.initial_demand
            
            shortage_amount = max(0, current_demand - current_supply)
            shortage_percentage = (shortage_amount / current_demand) * 100 if current_demand > 0 else 0
            
            period_data = TrajectoryPeriod(
                period=period,
                total_demand=current_demand,
                total_supply=current_supply,
                shortage_amount=shortage_amount,
                unsold=max(0, current_supply - current_demand),
                shortage_percentage=shortage_percentage,
                disrupted_manufacturers=scenario.disrupted_manufacturers if period < scenario.periods // 2 else [],
                fda_announcement="" if period > 0 else "FDA monitoring situation"
            )
            trajectory_periods.append(period_data)
        
        return SupplyTrajectory(
            periods=scenario.periods,
            scenario_summary={"fallback": True},
            trajectory=trajectory_periods,
            predicted_resolution_period=scenario.periods - 1,
            confidence_level="low",
            economic_reasoning={"error": "Fallback trajectory generated due to LLM failure"}
        )