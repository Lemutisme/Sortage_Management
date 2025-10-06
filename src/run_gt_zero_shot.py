#!/usr/bin/env python3
"""
Zero-Shot Ground Truth Experiments
==================================

Equivalent of run_gt_experiments() but using zero-shot trajectory prediction
instead of the full multi-agent simulation. Maintains same input/output format
for direct comparison with the agent-based results.
"""

import asyncio
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm
import uuid

from configs import SimulationConfig
from zero_shot_trajectory import ZeroShotTrajectoryPredictor, TrajectoryScenarioContext


def setup_logging():
    """Setup logging for zero-shot experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_trajectory_from_zero_shot(trajectory_data, scenario_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert zero-shot trajectory to agent simulation format."""
    
    market_trajectory = []
    
    for period_data in trajectory_data.trajectory:
        trajectory_period = {
            'period': period_data.period,
            'total_demand': period_data.total_demand,
            'total_supply': period_data.total_supply,
            'shortage_amount': period_data.shortage_amount,
            'unsold': period_data.unsold,
            'shortage_percentage': period_data.shortage_percentage,
            'disrupted_manufacturers': period_data.disrupted_manufacturers,
            'fda_announcement': period_data.fda_announcement
        }
        market_trajectory.append(trajectory_period)
    
    return market_trajectory


def calculate_summary_metrics(trajectory_data, market_trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary metrics compatible with agent simulation output."""
    
    shortage_percentages = [p['shortage_percentage'] for p in market_trajectory]
    shortage_periods = sum(1 for pct in shortage_percentages if pct > 0.001)
    
    # Find resolution time (first period where shortage < 5%)
    resolution_time = None
    for i, pct in enumerate(shortage_percentages):
        if pct < 5.0:  # Consider <5% as resolved
            resolution_time = i
            break
    
    # Calculate total costs/profits (simplified heuristic)
    total_demand = sum(p['total_demand'] for p in market_trajectory)
    total_supply = sum(p['total_supply'] for p in market_trajectory)
    total_shortage = sum(p['shortage_amount'] for p in market_trajectory)
    
    # Estimate buyer cost (demand * unit cost + shortage penalties)
    unit_cost = 1.0  # Assume base unit cost
    shortage_penalty = 1.1  # From config
    buyer_total_cost = total_demand * unit_cost + total_shortage * shortage_penalty
    
    # Estimate manufacturer profit (supply * unit profit - capacity costs)
    unit_profit = 1.0  # From config  
    capacity_cost = 0.5  # From config
    estimated_capacity_investment = max(0, total_supply - len(market_trajectory))  # Heuristic
    total_manufacturer_profit = total_supply * unit_profit - estimated_capacity_investment * capacity_cost
    
    return {
        'peak_shortage_percentage': max(shortage_percentages) / 100.0 if shortage_percentages else 0.0,
        'average_shortage_percentage': sum(shortage_percentages) / len(shortage_percentages) / 100.0 if shortage_percentages else 0.0,
        'total_shortage_periods': shortage_periods,
        'time_to_resolution': resolution_time,
        'total_manufacturer_profit': total_manufacturer_profit,
        'fda_intervention_rate': sum(1 for p in market_trajectory if p['fda_announcement']) / len(market_trajectory) if market_trajectory else 0.0
    }


def create_mock_fda_announcements(market_trajectory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create FDA announcements list from trajectory data."""
    
    announcements = []
    for period_data in market_trajectory:
        if period_data['fda_announcement']:
            announcements.append({
                'period': period_data['period'],
                'announcement_type': 'monitoring',  # Simplified
                'message': period_data['fda_announcement']
            })
    
    return announcements


async def run_single_zero_shot_experiment(
    row_dict: Dict[str, Any], 
    sim_number: int,
    predictor: ZeroShotTrajectoryPredictor
) -> Dict[str, Any]:
    """Run a single zero-shot experiment equivalent to agent simulation."""
    
    logger = logging.getLogger("ZeroShotGTExperiment")
    
    print(f"\n📊 Running Zero-Shot Ground Truth Experiment: gt_id_{row_dict['gt_id']}_simulation_{sim_number}")
    print("=" * 80)
    
    try:
        # Create scenario context matching agent simulation parameters
        disrupted_manufacturers = []
        if int(row_dict['disruption_number']) > 0:
            # Assume first manufacturer(s) are disrupted
            disrupted_manufacturers = list(range(int(row_dict['disruption_number'])))
        
        scenario = predictor.create_scenario_context(
            n_manufacturers=int(row_dict['n_manufacturers']),
            periods=int(row_dict['periods']),
            disruption_prob=0.05,  # Default from agent simulation
            disruption_magnitude=float(row_dict['disruption_magnitude']),
            disrupted_manufacturers=disrupted_manufacturers,
            initial_demand=1.0,  # Default baseline
        )
        
        # Calculate initial shortage percentage
        initial_shortage_pct = 0.0
        if disrupted_manufacturers:
            disrupted_capacity_loss = len(disrupted_manufacturers) * scenario.disruption_magnitude / scenario.n_manufacturers
            initial_supply = scenario.initial_demand * (1 - disrupted_capacity_loss)
            initial_shortage_pct = max(0, (scenario.initial_demand - initial_supply) / scenario.initial_demand * 100)
        
        # Make trajectory prediction
        trajectory_data = await predictor.predict_supply_trajectory(scenario)
        
        # Convert to agent simulation format
        market_trajectory = create_trajectory_from_zero_shot(trajectory_data, row_dict)
        summary_metrics = calculate_summary_metrics(trajectory_data, market_trajectory)
        fda_announcements = create_mock_fda_announcements(market_trajectory)
        
        # Create compatible results structure
        results = {
            'logging_session': {
                'simulation_id': str(uuid.uuid4())[:8],
                'method': 'zero_shot_trajectory'
            },
            'summary_metrics': summary_metrics,
            'market_trajectory': market_trajectory,
            'buyer_total_cost': sum(p['total_demand'] for p in market_trajectory) + sum(p['shortage_amount'] for p in market_trajectory) * 1.1,
            'fda_announcements': fda_announcements,
            'zero_shot_metadata': {
                'predicted_resolution_period': trajectory_data.predicted_resolution_period,
                'confidence_level': trajectory_data.confidence_level,
                'economic_reasoning': trajectory_data.economic_reasoning,
                'scenario_summary': trajectory_data.scenario_summary
            }
        }
        
        # Print summary matching agent simulation format
        print(f"\n📈 Zero-Shot Results:")
        metrics = results['summary_metrics']
        print(f"  Peak shortage: {metrics['peak_shortage_percentage']:.1%}")
        print(f"  Shortage periods: {metrics['total_shortage_periods']}/{scenario.periods}")
        print(f"  Total buyer cost: {results['buyer_total_cost']:.3f}")
        print(f"  Total manufacturer profit: {metrics['total_manufacturer_profit']:.3f}")
        print(f"  FDA interventions: {metrics['fda_intervention_rate']:.1%} of periods")
        print(f"  Predicted resolution: Period {trajectory_data.predicted_resolution_period}")
        print(f"  Confidence: {trajectory_data.confidence_level}")
        
        return results
        
    except Exception as e:
        logger.error(f"Zero-shot experiment failed: {e}")
        raise


async def run_gt_zero_shot_experiments(
    df: pd.DataFrame,
    show_progress: bool = True,
    export_dir: str = "zero_shot_evaluation",
    n_simulations: int = 1,
    model_override: str = None,
    provider_override: str = None
) -> pd.DataFrame:
    """
    Zero-shot equivalent of run_gt_experiments().
    
    Args:
        df: DataFrame with columns: gt_id, n_manufacturers, gt_type, periods, 
            disruption_number, disruption_magnitude
        show_progress: Whether to show progress bar
        export_dir: Directory to save results
        n_simulations: Number of simulations per scenario
        
    Returns:
        DataFrame with results in same format as agent simulation
    """
    
    setup_logging()
    logger = logging.getLogger("ZeroShotGTExperiments")
    
    # Setup export directory
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = export_path / f"gt_zero_shot_experiments_{ts}.csv"
    exists_header = False
    
    # Initialize zero-shot predictor
    config = SimulationConfig(
        llm_provider=provider_override or "openai",
        llm_model=model_override or "gpt-4o",
        llm_temperature=0.3
    )
    predictor = ZeroShotTrajectoryPredictor(config)
    
    logger.info(f"Starting zero-shot experiments on {len(df)} scenarios")
    logger.info(f"Using model: {config.llm_model}")
    logger.info(f"API key available: {config.api_key is not None}")
    
    iterator = tqdm(df.itertuples(index=False), total=len(df)) if show_progress else df.itertuples(index=False)
    comparative_results = []
    comparison_df = pd.DataFrame()
    
    for row in iterator:
        row_dict = row._asdict()
        temp_results = []
        
        for sim in range(n_simulations):
            try:
                # Run zero-shot experiment
                results = await run_single_zero_shot_experiment(row_dict, sim, predictor)
                
                # Extract key metrics for comparison (matching agent simulation format)
                metrics = results['summary_metrics']
                temp_results.append({
                    "scenario": "gt_id_" + str(row_dict['gt_id']),
                    "simulation_id": results['logging_session']['simulation_id'],
                    "#simulation": sim,
                    "gt_type": row_dict['gt_type'],
                    "n_manufacturers": int(row_dict['n_manufacturers']),
                    "periods": int(row_dict['periods']),
                    "disruption_prob": 0.05,  # Default used in agent simulation
                    "disruption_magnitude": row_dict['disruption_magnitude'],
                    "trajectory": results['market_trajectory'],
                    "peak_shortage": metrics['peak_shortage_percentage'],
                    "avg_shortage": metrics['average_shortage_percentage'],
                    "shortage_periods": metrics['total_shortage_periods'],
                    "resolution_time": metrics['time_to_resolution'],
                    "total_profit": metrics['total_manufacturer_profit'],
                    "buyer_cost": results['buyer_total_cost'],
                    "fda_interventions": len(results['fda_announcements']),
                    # Zero-shot specific fields
                    "predicted_resolution_period": results['zero_shot_metadata']['predicted_resolution_period'],
                    "confidence_level": results['zero_shot_metadata']['confidence_level'],
                    "method": "zero_shot_trajectory"
                })
                
                # Add trajectory data (flattened for CSV compatibility)
                # for i, period_data in enumerate(results['market_trajectory']):
                #     experiment_result.update({
                #         f"trajectory_period_{i}": period_data['period'],
                #         f"trajectory_total_demand_{i}": period_data['total_demand'],
                #         f"trajectory_total_supply_{i}": period_data['total_supply'],
                #         f"trajectory_shortage_amount_{i}": period_data['shortage_amount'],
                #         f"trajectory_unsold_{i}": period_data['unsold'],
                #         f"trajectory_shortage_percentage_{i}": period_data['shortage_percentage'],
                #         f"trajectory_disrupted_manufacturers_{i}": str(period_data['disrupted_manufacturers']),
                #         f"trajectory_fda_announcement_{i}": period_data['fda_announcement']
                #     })
                
                # temp_results.append(experiment_result)
                
            except Exception as e:
                logger.error(f"Scenario gt_id_{str(row_dict['gt_id'])}_simulation_{sim} failed: {e}")
                temp_results.append({
                    "scenario": "gt_id_" + str(row_dict['gt_id']),
                    "#simulation": sim,
                    "error": str(e),
                    "method": "zero_shot_trajectory"
                })
        
        # Save results incrementally
        if temp_results:
            temp_df = pd.DataFrame([r for r in temp_results if 'error' not in r])
            comparison_df = pd.concat([comparison_df, temp_df], ignore_index=True)
            comparative_results += temp_results
            
            if not temp_df.empty:
                if not exists_header:
                    temp_df.to_csv(csv_path, index=False)
                    exists_header = True
                else:
                    temp_df.to_csv(csv_path, mode='a', header=False, index=False)
                logger.info(f"Results saved to: {csv_path}")
    
    # Save final comprehensive results
    if not comparison_df.empty:
        final_csv_path = export_path / f"gt_zero_shot_complete_{ts}.csv"
        comparison_df.to_csv(final_csv_path, index=False)
        
        # Save detailed JSON results
        json_path = export_path / f"gt_zero_shot_detailed_{ts}.json"
        with open(json_path, 'w') as f:
            json.dump(comparative_results, f, indent=2, default=str)
        
        logger.info(f"Final results saved to:")
        logger.info(f"  - CSV: {final_csv_path}")
        logger.info(f"  - JSON: {json_path}")
        
        # Print summary statistics
        print(f"\n📊 Zero-Shot Experiments Summary:")
        print(f"=" * 60)
        print(f"• Total scenarios: {len(df)}")
        print(f"• Simulations per scenario: {n_simulations}")
        print(f"• Successful experiments: {len(comparison_df)}")
        print(f"• Failed experiments: {len(comparative_results) - len(comparison_df)}")
        
        if not comparison_df.empty:
            print(f"• Average peak shortage: {comparison_df['peak_shortage'].mean():.1%}")
            print(f"• Average resolution time: {comparison_df['resolution_time'].mean():.1f} periods")
            print(f"• Average FDA interventions: {comparison_df['fda_interventions'].mean():.1f}")
            
            # Group by market structure
            concentrated = comparison_df[comparison_df['n_manufacturers'] <= 3]
            competitive = comparison_df[comparison_df['n_manufacturers'] > 3]
            
            if not concentrated.empty:
                print(f"• Concentrated markets (≤3 mfg): {concentrated['resolution_time'].mean():.1f} avg resolution")
            if not competitive.empty:
                print(f"• Competitive markets (>3 mfg): {competitive['resolution_time'].mean():.1f} avg resolution")
    
    return comparison_df


async def main():
    """Main function to run zero-shot ground truth experiments."""
    
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Zero-Shot Ground Truth Experiments")
    parser.add_argument("--data", choices=["disc", "nodisc"], default="disc", 
                       help="Dataset to use: 'disc' for GT_Disc.csv, 'nodisc' for GT_NoDisc.csv")
    parser.add_argument("--n_simulations", type=int, default=3, 
                       help="Number of simulations per scenario")
    parser.add_argument("--export-dir", default="gt_evaluation_zero_shot",
                       help="Directory to save results")
    parser.add_argument("--max-scenarios", type=int, help="Maximum number of scenarios to run")
    parser.add_argument("--model", type=str, help="Override LLM model")
    parser.add_argument("--provider", type=str, help="Override LLM provider")
    args = parser.parse_args()
    
    print("🧬 Zero-Shot Ground Truth Experiments")
    print("=" * 70)
    
    # Load data
    path = Path(__file__).resolve().parent
    if args.data == "disc":
        csv_path = path / "../data/GT_Disc.csv"
        print("Using GT_Disc.csv (disruption scenarios)")
    else:
        csv_path = path / "../data/GT_NoDisc.csv"
        print("Using GT_NoDisc.csv (no disruption scenarios)")
    
    df = pd.read_csv(csv_path)
    
    if args.max_scenarios:
        df = df.head(args.max_scenarios)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Running {args.n_simulations} simulations per scenario")
    print(f"Total experiments: {len(df) * args.n_simulations}")
    
    if args.model:
        args.export_dir += f"/model_{args.model}"
    # Run experiments
    results_df = await run_gt_zero_shot_experiments(
        df, 
        show_progress=True,
        export_dir=args.export_dir,
        n_simulations=args.n_simulations,
        model_override=args.model,
        provider_override=args.provider
    )
    
    print(f"\n✅ Zero-shot experiments completed!")
    print(f"Results shape: {results_df.shape}")
    print(f"💾 Results saved to: {args.export_dir}/")


if __name__ == "__main__":
    asyncio.run(main())