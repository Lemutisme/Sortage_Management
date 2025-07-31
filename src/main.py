"""
Drug Shortage Multi-Agent Simulation System with Comprehensive Logging
====================================================================

A comprehensive simulation framework for modeling pharmaceutical supply chain
dynamics with LLM-based agents and detailed logging for analysis.
"""

from typing import Dict, Any
import pandas as pd
import asyncio
import os
from pathlib import Path
from datetime import datetime

from simulator import SimulationCoordinator
from configs import SimulationConfig
from tqdm import tqdm

# =============================================================================
# Enhanced Example Usage with Logging
# =============================================================================

async def run_logged_simulation(config: SimulationConfig, 
                                start_with_disruption: bool = False,
                                simulation_name: str = "default") -> Dict[str, Any]:
    """
    Run a simulation with comprehensive logging and analysis.
    
    Args:
        config: Simulation configuration
        simulation_name: Descriptive name for this simulation run
        
    Returns:
        Complete simulation results with logging metadata
    """
    
    print(f"\n🚀 Starting simulation: {simulation_name}")
    print(f"Configuration: {config.n_manufacturers} manufacturers, {config.n_periods} periods")
    
    # Create and run simulation with logging
    coordinator = SimulationCoordinator(config)
    
    try:
        results = await coordinator.run_simulation(start_with_disruption)
        
        # Print logging summary
        logging_summary = coordinator.get_logging_summary()
        print(f"\n📊 Logging Summary:")
        print(f"  Simulation ID: {logging_summary['simulation_id']}")
        print(f"  Log Directory: {logging_summary['session_directory']}")
        print(f"  Total Events: {logging_summary['logging_stats']['total_events']}")
        print(f"  Decision Events: {logging_summary['logging_stats']['decision_events']}")
        print(f"  LLM Calls: {logging_summary['logging_stats']['llm_calls']}")
        
        # Print simulation results summary
        print(f"\n📈 Simulation Results:")
        metrics = results['summary_metrics']
        print(f"  Peak shortage: {metrics['peak_shortage_percentage']:.1%}")
        print(f"  Shortage periods: {metrics['total_shortage_periods']}/{config.n_periods}")
        print(f"  Total buyer cost: {results['buyer_total_cost']:.3f}")
        print(f"  Total manufacturer profit: {metrics['total_manufacturer_profit']:.3f}")
        print(f"  FDA interventions: {metrics['fda_intervention_rate']:.1%} of periods")
        # print(f"  Peak shortage: {metrics['peak_shortage_percentage']}")
        # print(f"  Shortage periods: {metrics['total_shortage_periods']}/{config.n_periods}")
        # print(f"  Total buyer cost: {results['buyer_total_cost']}")
        # print(f"  Total manufacturer profit: {metrics['total_manufacturer_profit']}")
        # print(f"  FDA interventions: {metrics['fda_intervention_rate']} of periods")
        
        return results
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        raise

async def run_comparative_study():
    """Run multiple simulations to compare different scenarios."""
    
    print("🔬 Running Comparative Study with Different Market Conditions")
    
    scenarios = [
        {
            "name": "Low Disruption Baseline",
            "config": SimulationConfig(
                n_manufacturers=4,
                n_periods=4,
                disruption_probability=0.02,
                disruption_magnitude=0.15
            )
        },
        {
            "name": "High Disruption Stress Test", 
            "config": SimulationConfig(
                n_manufacturers=4,
                n_periods=4,
                disruption_probability=0.10,
                disruption_magnitude=0.30
            )
        },
        {
            "name": "Concentrated Market (3 manufacturers)",
            "config": SimulationConfig(
                n_manufacturers=3,
                n_periods=4,
                disruption_probability=0.05,
                disruption_magnitude=0.20
            )
        },
        {
            "name": "Competitive Market (6 manufacturers)",
            "config": SimulationConfig(
                n_manufacturers=6,
                n_periods=4,
                disruption_probability=0.05,
                disruption_magnitude=0.20
            )
        }
    ]
    
    comparative_results = []
    
    for scenario in scenarios:
        print(f"\n--- Running Scenario: {scenario['name']} ---")
        
        try:
            results = await run_logged_simulation(
                scenario['config'], 
                scenario['name']
            )
            
            # Extract key metrics for comparison
            metrics = results['summary_metrics']
            comparative_results.append({
                "scenario": scenario['name'],
                "n_manufacturers": scenario['config'].n_manufacturers,
                "disruption_prob": scenario['config'].disruption_probability,
                "disruption_magnitude": scenario['config'].disruption_magnitude,
                "peak_shortage": metrics['peak_shortage_percentage'],
                "avg_shortage": metrics['average_shortage_percentage'],
                "shortage_periods": metrics['total_shortage_periods'],
                "resolution_time": metrics['time_to_resolution'],
                "total_profit": metrics['total_manufacturer_profit'],
                "buyer_cost": results['buyer_total_cost'],
                "fda_interventions": len(results['fda_announcements']),
                "simulation_id": results['logging_session']['simulation_id']
            })
            
        except Exception as e:
            print(f"❌ Scenario '{scenario['name']}' failed: {e}")
            comparative_results.append({
                "scenario": scenario['name'],
                "error": str(e)
            })
    
    # Create comparison summary
    if comparative_results:
        comparison_df = pd.DataFrame([r for r in comparative_results if 'error' not in r])
        
        if not comparison_df.empty:
            print(f"\n📊 Comparative Analysis Summary:")
            print("=" * 80)
            
            for _, row in comparison_df.iterrows():
                print(f"{row['scenario']:<30} | "
                      f"Peak: {row['peak_shortage']:.1%} | "
                      f"Periods: {row['shortage_periods']}/4 | "
                      f"Resolution: {row['resolution_time'] or 'N/A'} | "
                      f"FDA: {row['fda_interventions']}")
            
            # Save comparison results
            comparison_df.to_csv("comparative_study_results.csv", index=False)
            print(f"\n💾 Comparative results saved to: comparative_study_results.csv")
        
        return comparative_results

async def analyze_agent_decisions(simulation_results: Dict[str, Any]):
    """Analyze agent decision patterns from simulation results."""
    
    print(f"\n🧠 Analyzing Agent Decision Patterns")
    
    # Manufacturer decision analysis
    manufacturer_decisions = simulation_results['decision_history']['manufacturers']
    
    print(f"\n📊 Manufacturer Decision Analysis:")
    for i, decisions in enumerate(manufacturer_decisions):
        if decisions:
            investments = []
            confidences = []
            
            for decision in decisions:
                final_decision = decision.get('final_decision', {})
                investment = float(final_decision.get('decision', {}).get('capacity_investment', 0))
                confidence = final_decision.get('confidence', 'unknown')
                
                investments.append(investment)
                confidences.append(confidence)
            
            total_investment = sum(investments)
            avg_confidence = len([c for c in confidences if c in ['moderate', 'high']]) / len(confidences) if confidences else 0
            
            print(f"  Manufacturer {i}: Total Investment: {total_investment:.3f}, Confidence Rate: {avg_confidence:.1%}")
    
    # Buyer decision analysis
    buyer_decisions = simulation_results['decision_history']['buyer']
    
    if buyer_decisions:
        print(f"\n🏥 Buyer Decision Analysis:")
        
        demand_quantities = []
        stockpiling_periods = 0
        
        for decision in buyer_decisions:
            final_decision = decision.get('final_decision', {})
            demand = final_decision.get('decision', {}).get('demand_quantity', 1.0)
            rationale = final_decision.get('decision', {}).get('demand_rationale', 'baseline')
            
            demand_quantities.append(demand)
            
            if 'stockpile' in rationale.lower() or demand > 1.2:
                stockpiling_periods += 1
        
        avg_demand = sum(demand_quantities) / len(demand_quantities) if demand_quantities else 1.0
        max_demand = max(demand_quantities) if demand_quantities else 1.0
        
        print(f"  Average Demand: {avg_demand:.3f} ({avg_demand:.1%} of baseline)")
        print(f"  Peak Demand: {max_demand:.3f}")
        print(f"  Stockpiling Periods: {stockpiling_periods}/{len(buyer_decisions)}")
    
    # FDA decision analysis
    fda_decisions = simulation_results['decision_history']['fda']
    
    if fda_decisions:
        print(f"\n🏛️ FDA Decision Analysis:")
        
        announcement_types = {}
        for decision in fda_decisions:
            final_decision = decision.get('final_decision', {})
            announcement_type = final_decision.get('decision', {}).get('announcement_type', 'none')
            announcement_types[announcement_type] = announcement_types.get(announcement_type, 0) + 1
        
        for ann_type, count in announcement_types.items():
            percentage = count / len(fda_decisions) * 100
            print(f"  {ann_type.title()}: {count} times ({percentage:.1f}%)")

def export_detailed_analysis(results: Dict[str, Any], export_dir: str = "analysis_exports"):
    """Export detailed analysis files for further research."""
    
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)
    
    simulation_id = results['logging_session']['simulation_id']
    
    print(f"\n💾 Exporting detailed analysis to {export_dir}/")
    
    # Export market trajectory
    market_df = pd.DataFrame(results['market_trajectory'])
    market_df.to_csv(export_path / f"market_trajectory_{simulation_id}.csv", index=False)
    
    # Export manufacturer states
    manufacturer_df = pd.DataFrame(results['manufacturer_summaries'])
    manufacturer_df.to_csv(export_path / f"manufacturer_states_{simulation_id}.csv", index=False)
    
    # Export buyer summary
    buyer_summary = results['buyer_summary']
    if isinstance(buyer_summary, dict) and 'status' not in buyer_summary:
        buyer_df = pd.DataFrame([buyer_summary])
        buyer_df.to_csv(export_path / f"buyer_summary_{simulation_id}.csv", index=False)
    
    # Export FDA announcements
    if results['fda_announcements']:
        fda_df = pd.DataFrame(results['fda_announcements'])
        fda_df.to_csv(export_path / f"fda_announcements_{simulation_id}.csv", index=False)
    
    print(f"  ✅ Analysis files exported with ID: {simulation_id}")

# =============================================================================
# Main Execution Functions
# =============================================================================

async def run_single_example(start_with_disruption: bool = False):
    """Run a single example simulation with detailed logging."""
    
    config = SimulationConfig(
        n_manufacturers=2,
        n_periods=4,
        disruption_probability=0.05,
        disruption_magnitude=0.3,
        llm_temperature=0.3,
        n_disruptions_if_forced_disruption=1
        # Uncomment and set your API key:
        # api_key=os.getenv("OPENAI_API_KEY")
    )

    results = await run_logged_simulation(config, start_with_disruption, "Single Example Run")

    # Analyze agent decisions
    await analyze_agent_decisions(results)
    
    # Export detailed analysis
    export_detailed_analysis(results)
    
    return results

async def run_gt_experiments(
        df: pd.DataFrame,
        show_progress: bool = True,
        export_dir: str = "gt_evaluation",
        n_simulations: int = 3
):
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = export_path / f"gt_experiments_{ts}.csv"
    exists_header = False

    iterator = tqdm(df.itertuples(index=False), total=len(df)) if show_progress else df.itertuples(index=False)
    comparative_results = []
    comparison_df = pd.DataFrame()
    for row in iterator:
        row_dict = row._asdict()
        temp_results = []
        for sim in range(n_simulations):
            print(f"\n📊 Running Ground Truth Experiment: gt_id_{row_dict['gt_id']}_simulation_{sim}")
            print("=" * 80)
            try:
                config = SimulationConfig(
                    n_manufacturers=int(row_dict['n_manufacturers']),
                    n_periods=int(row_dict['periods']),
                    disruption_probability=0.05,
                    disruption_magnitude=row_dict['disruption_magnitude'],
                    llm_temperature=0.3,
                    n_disruptions_if_forced_disruption=int(row_dict['disruption_number'])
                )
                
                start_with_disruption = True if row_dict['disruption_number'] > 0 else False

                results = await run_logged_simulation(config, start_with_disruption, "gt_id_" + str(row_dict['gt_id']))
                # Analyze agent decisions
                await analyze_agent_decisions(results)
                # Export detailed analysis
                export_detailed_analysis(results)

                # Extract key metrics for comparison
                metrics = results['summary_metrics']
                temp_results.append({
                    "scenario": "gt_id_" + str(row_dict['gt_id']),
                    "simulation_id": results['logging_session']['simulation_id'],
                    "#simulation": sim,
                    "gt_type":row_dict['gt_type'],
                    "n_manufacturers": config.n_manufacturers,
                    "periods":row_dict['periods'],
                    "disruption_prob": config.disruption_probability,
                    "disruption_magnitude": config.disruption_magnitude,
                    "trajectory": results['market_trajectory'],
                    "peak_shortage": metrics['peak_shortage_percentage'],
                    "avg_shortage": metrics['average_shortage_percentage'],
                    "shortage_periods": metrics['total_shortage_periods'],
                    "resolution_time": metrics['time_to_resolution'],
                    "total_profit": metrics['total_manufacturer_profit'],
                    "buyer_cost": results['buyer_total_cost'],
                    "fda_interventions": len(results['fda_announcements']),
                })
            except Exception as e:
                print(f"❌ Scenario gt_id_{str(row_dict['gt_id'])}_simulation_{sim} failed: {e}")
                comparative_results.append({
                    "scenario": "gt_id_" + str(row_dict['gt_id']),
                    "error": str(e)
                })
    
        if temp_results:
            temp_df = pd.DataFrame([r for r in temp_results if 'error' not in r])
            comparison_df = pd.concat([comparison_df, temp_df], ignore_index=True) 
            comparative_results+=temp_results
        
            if not temp_df.empty:
                if not exists_header:
                    temp_df.to_csv(csv_path, index=False)
                    exists_header = True
                else:
                    temp_df.to_csv(csv_path, mode='a', header=False, index=False)
                print(f"\n💾 Comparative results saved to: gt_experiments_{ts}.csv")
    return comparison_df


async def run_quick_policy_test():
    """Quick test of different policy scenarios."""
    
    print("🎯 Quick Policy Effectiveness Test")
    
    # This would require implementing proactive FDA mode
    # For now, we test with different disruption scenarios
    
    low_disruption_config = SimulationConfig(
        n_manufacturers=4,
        n_periods=4,
        disruption_probability=0.02,
        disruption_magnitude=0.10
    )
    
    high_disruption_config = SimulationConfig(
        n_manufacturers=4,
        n_periods=4,
        disruption_probability=0.08,
        disruption_magnitude=0.25
    )
    
    print("\n--- Low Disruption Environment ---")
    low_results = await run_logged_simulation(low_disruption_config, "Low Disruption Policy Test")
    
    print("\n--- High Disruption Environment ---")
    high_results = await run_logged_simulation(high_disruption_config, "High Disruption Policy Test")
    
    # Compare outcomes
    print(f"\n📊 Policy Test Comparison:")
    print(f"Low Disruption - Peak Shortage: {low_results['summary_metrics']['peak_shortage_percentage']:.1%}")
    print(f"High Disruption - Peak Shortage: {high_results['summary_metrics']['peak_shortage_percentage']:.1%}")
    
    print(f"Low Disruption - FDA Interventions: {len(low_results['fda_announcements'])}")
    print(f"High Disruption - FDA Interventions: {len(high_results['fda_announcements'])}")

if __name__ == "__main__":
    import sys
    
    print("🧬 Drug Shortage Multi-Agent Simulation with Comprehensive Logging")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "comparative":
            print("Running comparative study...")
            asyncio.run(run_comparative_study())
        elif mode == "policy":
            print("Running policy test...")
            asyncio.run(run_quick_policy_test())
        elif mode == "gt_experiment_dic":
            print("Running ground truth experiment...")
            HERE = Path(__file__).resolve().parent
            csv_path = HERE/"../data"/"GT_Disc.csv"
            csv_path = HERE/"../data"/"GT_Disc_tester.csv"
            df = pd.read_csv(csv_path)
            print(df.shape)
            asyncio.run(run_gt_experiments(df))
        elif mode == "gt_experiment_nodic":
            print("Running ground truth experiment (No discontinued)...")
            HERE = Path(__file__).resolve().parent
            csv_path = HERE/"../data"/"GT_NoDisc.csv"
            df = pd.read_csv(csv_path)
            print(df.shape)
            asyncio.run(run_gt_experiments(df))
        else:
            print("Unknown mode. Running single example...")
            asyncio.run(run_single_example())
    else:
        print("Running single example simulation...")
        asyncio.run(run_single_example(start_with_disruption=True))
    
    print("\n✅ Simulation completed! Check the generated log files for detailed analysis.")
    print("💡 Log files include:")
    print("   - Comprehensive event logs (JSONL format)")
    print("   - Agent-specific decision logs")
    print("   - Market outcome tracking")
    print("   - LLM call logs for debugging")
    print("   - CSV exports for analysis")