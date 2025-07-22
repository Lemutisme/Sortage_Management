import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid

from configs import SimulationConfig, MarketState, DisruptionEvent


@dataclass
class LogEvent:
    """Base class for all simulation log events."""
    timestamp: str
    simulation_id: str
    period: int
    event_type: str
    event_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class DecisionLogEvent(LogEvent):
    """Specialized log event for agent decisions."""
    agent_id: str
    agent_type: str
    decision_stage: str  # 'analyze' or 'decide'
    llm_calls: List[Dict[str, Any]]
    reasoning: Optional[str] = None
    confidence: Optional[str] = None


@dataclass
class MarketLogEvent(LogEvent):
    """Specialized log event for market outcomes."""
    demand: float
    supply: float
    shortage: float
    shortage_percentage: float
    disrupted_manufacturers: List[int]
    allocations: List[float]


class SimulationLogger:
    """Comprehensive logging system for the drug shortage simulation."""
    
    def __init__(self, config: SimulationConfig, log_dir: str = "simulation_logs"):
        self.config = config
        self.simulation_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        
        # Create log directory
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Log files
        self.session_dir = self.log_dir / f"sim_{self.simulation_id}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize structured event storage
        self.events: List[LogEvent] = []
        self.decision_events: List[DecisionLogEvent] = []
        self.market_events: List[MarketLogEvent] = []
        
        # Setup file handlers
        self._setup_loggers()
        
        # Log simulation start
        self.log_simulation_start()
    
    def _setup_loggers(self):
        """Setup different log handlers for different types of information."""
        
        # Main simulation logger
        self.main_logger = logging.getLogger(f"Simulation_{self.simulation_id}")
        self.main_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.main_logger.handlers.clear()
        
        # Console handler for real-time monitoring
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.main_logger.addHandler(console_handler)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.session_dir / "simulation.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.main_logger.addHandler(file_handler)
        
        # JSON structured log file
        self.json_log_file = open(self.session_dir / "events.jsonl", "w")
        
        # Agent-specific loggers
        self.agent_loggers = {}
    
    def get_agent_logger(self, agent_id: str) -> logging.Logger:
        """Get or create agent-specific logger."""
        if agent_id not in self.agent_loggers:
            logger = logging.getLogger(f"Agent_{agent_id}_{self.simulation_id}")
            logger.setLevel(logging.DEBUG)
            
            # Agent-specific file handler
            agent_file = self.session_dir / f"agent_{agent_id}.log"
            handler = logging.FileHandler(agent_file)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            self.agent_loggers[agent_id] = logger
        
        return self.agent_loggers[agent_id]
    
    def log_simulation_start(self):
        """Log simulation initialization."""
        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            simulation_id=self.simulation_id,
            period=-1,
            event_type="simulation_start",
            event_data={
                "config": asdict(self.config),
                "session_directory": str(self.session_dir)
            }
        )
        self._record_event(event)
        self.main_logger.info(f"Simulation {self.simulation_id} started with config: {self.config}")

    def log_period_start(self, period: int):
        """Log the start of a new period."""
        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            simulation_id=self.simulation_id,
            period=period,
            event_type="period_start",
            event_data={"period": period}
        )
        self._record_event(event)
        self.main_logger.info(f"=== Period {period + 1}/{self.config.n_periods} Started ===")

    def log_disruption_events(self, period: int, new_disruptions: List[DisruptionEvent], 
                            active_disruptions: List[DisruptionEvent]):
        """Log disruption events and status."""
        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            simulation_id=self.simulation_id,
            period=period,
            event_type="disruptions",
            event_data={
                "new_disruptions": [asdict(d) for d in new_disruptions],
                "active_disruptions": [asdict(d) for d in active_disruptions],
                "total_disrupted": len(active_disruptions)
            }
        )
        self._record_event(event)

        if new_disruptions:
            for disruption in new_disruptions:
                self.main_logger.warning(
                    f"New disruption: Manufacturer {disruption.manufacturer_id}, "
                    f"duration {disruption.duration}, magnitude {disruption.magnitude:.1%}"
                )

    def log_agent_decision(self, agent_id: str, agent_type: str, period: int, 
                          decision_stage: str, context: Dict[str, Any], 
                          result: Dict[str, Any], llm_calls: List[Dict[str, Any]] = None):
        """Log agent decision-making process."""

        # Extract reasoning and confidence if available
        reasoning = result.get("reasoning", {})
        confidence = result.get("confidence", "unknown")

        decision_event = DecisionLogEvent(
            timestamp=datetime.now().isoformat(),
            simulation_id=self.simulation_id,
            period=period,
            event_type="agent_decision",
            event_data={
                "context": context,
                "result": result,
                "decision_stage": decision_stage
            },
            agent_id=agent_id,
            agent_type=agent_type,
            decision_stage=decision_stage,
            llm_calls=llm_calls or [],
            reasoning=json.dumps(reasoning) if reasoning else None,
            confidence=confidence
        )

        self._record_event(decision_event)
        self.decision_events.append(decision_event)

        # Agent-specific logging
        agent_logger = self.get_agent_logger(agent_id)
        agent_logger.info(f"Decision ({decision_stage}): {json.dumps(result, indent=2)}")

        # Main simulation log
        decision_summary = result.get("decision", {})
        self.main_logger.info(
            f"{agent_type} {agent_id} decision ({decision_stage}): "
            f"{json.dumps(decision_summary)} [confidence: {confidence}]"
        )

    def log_llm_call(self, agent_id: str, period: int, stage: str, 
                    system_prompt: str, user_prompt: str, response: str, 
                    success: bool, attempt: int = 1, error: str = None):
        """Log LLM API calls for debugging and analysis."""
        llm_event = LogEvent(
            timestamp=datetime.now().isoformat(),
            simulation_id=self.simulation_id,
            period=period,
            event_type="llm_call",
            event_data={
                "agent_id": agent_id,
                "stage": stage,
                "system_prompt": system_prompt[:200] + "..." if len(system_prompt) > 200 else system_prompt,
                "user_prompt": user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt,
                "response": response[:1000] + "..." if len(response) > 1000 else response,
                "success": success,
                "attempt": attempt,
                "error": error,
                "response_length": len(response) if response else 0
            }
        )
        self._record_event(llm_event)

        agent_logger = self.get_agent_logger(agent_id)
        if success:
            agent_logger.debug(f"LLM call successful (attempt {attempt}): {response[:100]}...")
        else:
            agent_logger.error(f"LLM call failed (attempt {attempt}): {error}")

    def log_market_outcome(self, period: int, market_state: MarketState, 
                          allocations: List[float], manufacturer_states: List[Dict]):
        """Log market clearing and allocation results."""
        market_event = MarketLogEvent(
            timestamp=datetime.now().isoformat(),
            simulation_id=self.simulation_id,
            period=period,
            event_type="market_outcome",
            event_data={
                "market_state": asdict(market_state),
                "allocations": allocations,
                "manufacturer_states": manufacturer_states
            },
            demand=market_state.total_demand,
            supply=market_state.total_supply,
            shortage=market_state.shortage_amount,
            shortage_percentage=market_state.shortage_percentage,
            disrupted_manufacturers=market_state.disrupted_manufacturers,
            allocations=allocations
        )

        self._record_event(market_event)
        self.market_events.append(market_event)
        
        self.main_logger.info(
            f"Market outcome - Demand: {market_state.total_demand:.3f}, "
            f"Supply: {market_state.total_supply:.3f}, "
            f"Shortage: {market_state.shortage_amount:.3f} ({market_state.shortage_percentage:.1%})"
        )

        if market_state.fda_announcement:
            self.main_logger.info(f"FDA Announcement: {market_state.fda_announcement}")

    def log_simulation_end(self, results: Dict[str, Any]):
        """Log simulation completion and final results."""
        duration = (datetime.now() - self.start_time).total_seconds()

        event = LogEvent(
            timestamp=datetime.now().isoformat(),
            simulation_id=self.simulation_id,
            period=self.config.n_periods,
            event_type="simulation_end",
            event_data={
                "duration_seconds": duration,
                "summary_metrics": results.get("summary_metrics", {}),
                "final_states": {
                    "manufacturer_states": results.get("manufacturer_states", []),
                    "buyer_total_cost": results.get("buyer_total_cost", 0),
                    "fda_announcements": results.get("fda_announcements", [])
                }
            }
        )
        self._record_event(event)

        self.main_logger.info(f"Simulation {self.simulation_id} completed in {duration:.1f} seconds")
        self.main_logger.info(f"Final metrics: {results.get('summary_metrics', {})}")

        # Export analysis files
        self.export_analysis_files(results)

    def _record_event(self, event: LogEvent):
        """Record event to all appropriate storage."""
        self.events.append(event)

        # Write to JSON lines file
        self.json_log_file.write(json.dumps(event.to_dict()) + "\n")
        self.json_log_file.flush()

    def export_analysis_files(self, results: Dict[str, Any]):
        """Export data in formats suitable for analysis."""

        # 1. Export events as CSV for easy analysis
        if self.events:
            events_df = pd.DataFrame([event.to_dict() for event in self.events])
            events_df.to_csv(self.session_dir / "all_events.csv", index=False)

        # 2. Export decision events separately
        if self.decision_events:
            decisions_df = pd.DataFrame([event.to_dict() for event in self.decision_events])
            decisions_df.to_csv(self.session_dir / "decisions.csv", index=False)

        # 3. Export market events
        if self.market_events:
            market_df = pd.DataFrame([event.to_dict() for event in self.market_events])
            market_df.to_csv(self.session_dir / "market_outcomes.csv", index=False)

        # 4. Export complete simulation results
        with open(self.session_dir / "simulation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # 5. Create summary report
        self._create_summary_report(results)

        self.main_logger.info(f"Analysis files exported to {self.session_dir}")

    def _create_summary_report(self, results: Dict[str, Any]):
        """Create a human-readable summary report."""

        report_lines = [
            f"Drug Shortage Simulation Report",
            f"{'=' * 40}",
            f"Simulation ID: {self.simulation_id}",
            f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {(datetime.now() - self.start_time).total_seconds():.1f} seconds",
            f"",
            f"Configuration:",
            f"  Manufacturers: {self.config.n_manufacturers}",
            f"  Periods: {self.config.n_periods}",
            f"  Disruption Probability: {self.config.disruption_probability:.1%}",
            f"  Disruption Magnitude: {self.config.disruption_magnitude:.1%}",
            f"",
            f"Results:",
        ]

        metrics = results.get("summary_metrics", {})
        report_lines.extend([
            f"  Peak Shortage: {metrics.get('peak_shortage_percentage', 0):.1%}",
            f"  Shortage Periods: {metrics.get('total_shortage_periods', 0)}/{self.config.n_periods}",
            f"  Average Shortage: {metrics.get('average_shortage_percentage', 0):.1%}",
            f"  Time to Resolution: {metrics.get('time_to_resolution', 'Not resolved')}",
            f"  Total Buyer Cost: {results.get('buyer_total_cost', 0):.3f}",
            f"  Total Manufacturer Profit: {metrics.get('total_manufacturer_profit', 0):.3f}",
            f"",
            f"Events Summary:",
            f"  Total Events: {len(self.events)}",
            f"  Decision Events: {len(self.decision_events)}",
            f"  Market Events: {len(self.market_events)}",
            f"  LLM Calls: {len([e for e in self.events if e.event_type == 'llm_call'])}",
        ])

        # Write summary report
        with open(self.session_dir / "summary_report.txt", "w") as f:
            f.write("\n".join(report_lines))

    def get_analysis_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Return analysis-ready dataframes."""
        return {
            "all_events": pd.DataFrame([event.to_dict() for event in self.events]),
            "decisions": pd.DataFrame([event.to_dict() for event in self.decision_events]),
            "market_outcomes": pd.DataFrame([event.to_dict() for event in self.market_events]),
        }

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'json_log_file'):
            self.json_log_file.close()
        
        # Close all file handlers
        for handler in self.main_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.main_logger.removeHandler(handler)
        
        for agent_logger in self.agent_loggers.values():
            for handler in agent_logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    agent_logger.removeHandler(handler)


class LoggerMixin:
    """Mixin class to add logging capabilities to agents."""
    
    def setup_logger(self, simulation_logger: SimulationLogger):
        """Setup logger for this agent."""
        self.simulation_logger = simulation_logger
        self.agent_logger = simulation_logger.get_agent_logger(self.agent_id)

    def log_decision(self, period: int, decision_stage: str, context: Dict[str, Any], 
                    result: Dict[str, Any], llm_calls: List[Dict[str, Any]] = None):
        """Log agent decision."""
        if hasattr(self, 'simulation_logger'):
            self.simulation_logger.log_agent_decision(
                self.agent_id, self.__class__.__name__, period, 
                decision_stage, context, result, llm_calls
            )

    def log_llm_call(self, period: int, stage: str, system_prompt: str, 
                    user_prompt: str, response: str, success: bool, 
                    attempt: int = 1, error: str = None):
        """Log LLM call."""
        if hasattr(self, 'simulation_logger'):
            self.simulation_logger.log_llm_call(
                self.agent_id, period, stage, system_prompt, 
                user_prompt, response, success, attempt, error
            )