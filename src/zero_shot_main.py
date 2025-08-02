from zero_shot_trajectory import ZeroShotTrajectoryPredictor
from configs import SimulationConfig
import asyncio

# Setup
config = SimulationConfig(llm_model="gpt-4o")
predictor = ZeroShotTrajectoryPredictor(config)

# Create scenario
scenario = predictor.create_scenario_context(
    n_manufacturers=4,
    periods=5,
    disruption_prob=0.05,
    disruption_magnitude=0.4,
    disrupted_manufacturers=[0],
    initial_demand=1.0,
    initial_supply=0.9
)

# Predict trajectory
async def main():
    trajectory = await predictor.predict_supply_trajectory(scenario)

    # Access results
    for period in trajectory.trajectory:
        print(f"Period {period.period}:     shortage: {period.shortage_percentage:.1f}% shortage")
        print(f"Period {period.period}:     total_demand: {period.total_demand:.2f}")
        print(f"Period {period.period}:     total_supply: {period.total_supply:.2f}")
        print(f"Period {period.period}:     shortage_amount: {period.shortage_amount:.2f}")
        print(f"Period {period.period}:     unsold: {period.unsold:.2f}")
        print(f"Period {period.period}:     shortage_percentage: {period.shortage_percentage:.2f}")
        print(f"Period {period.period}:     disrupted_manufacturers: {period.disrupted_manufacturers}")
        print(f"Period {period.period}:     fda_announcement: {period.fda_announcement}")

asyncio.run(main())