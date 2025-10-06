import pandas as pd
import ast

# --------------------------------------------------------------------------- #
# 1. FDA-intervention proportion (FIP)
# --------------------------------------------------------------------------- #
def compute_fip(df: pd.DataFrame) -> float:
    """
    FDA-intervention proportion (FIP).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns
        - 'fda_interventions' : int  – number of FDA actions in the run
        - 'periods'           : int  – length of the run

    Returns
    -------
    float
        Average share of periods in which FDA intervenes, across all runs.
    """
    return (df["fda_interventions"] / df["periods"]).mean()


# --------------------------------------------------------------------------- #
# 2. Mean resolution-lag percentage (RLP)
# --------------------------------------------------------------------------- #
def compute_mean_rlp(
    df: pd.DataFrame,
    *,
    shortage_eps: float = 1e-3,
    actual_demand: float = 1.0,
) -> float:
    """
    Mean resolution-lag percentage (RLP).

    Parameters
    ----------
    df : pd.DataFrame
        Expected columns
        - 'scenario'   : identifier of each simulated run
        - 'trajectory' : list[dict] holding period-level stats.  Each dict needs
                         keys 'period', 'total_supply'.  Additional keys are OK.
    shortage_eps : float, optional
        Tolerance below the supply target that still counts as a shortage.
    supply_target : float, optional
        The demand level viewed as “fully met” (default = 1.0).

    Returns
    -------
    float
        Mean RLP across all scenarios.
    """
    # --- 1.  Flatten the list-of-dict trajectories -------------------------- #
    tmp = df.explode("trajectory", ignore_index=True)
    traj_cols = (
        pd.json_normalize(tmp["trajectory"])
          .rename(lambda c: f"trajectory_{c}", axis=1)
    )
    flat = (
        tmp.drop(columns="trajectory")
           .reset_index(drop=True)
           .join(traj_cols)
    )

    # --- 2.  Shortage flag --------------------------------------------------- #
    flat["shortage_flag"] = (
        flat["trajectory_total_supply"] - actual_demand <= -shortage_eps
    )

    # --- 3.  Earliest permanently clear period per scenario ----------------- #
    def _earliest_sustained_clear(g: pd.DataFrame) -> int:
        g = g.sort_values("trajectory_period").reset_index(drop=True)

        # shortage anywhere *after* current period
        future_shortage = (
            g["shortage_flag"].shift(-1, fill_value=False)[::-1]
              .cummax()[::-1]
        )
        clear = g[(~g["shortage_flag"]) & (~future_shortage)]

        if clear.empty:                         # never clears
            return int(g["trajectory_period"].max() + 1)
        return int(clear["trajectory_period"].iloc[0])

    t_sim = (
        flat.groupby("scenario", sort=False)
            .apply(_earliest_sustained_clear)
            .rename("t_sim")
    )

    # --- 4.  Ground-truth resolution time ----------------------------------- #
    t_gt = (
        flat.groupby("scenario")["trajectory_period"]
            .max()
            .rename("t_gt")
    )

    # --- 5.  Scenario horizon (for runs that never clear) ------------------- #
    T = (
        flat.groupby("scenario")["trajectory_period"]
            .max()
            .rename("T")
    )

    meta = (
        t_sim.to_frame()
             .join(T, how="left")
             .reset_index()
             .merge(t_gt.reset_index(), on="scenario", how="left")
    )

    # add 1 so that periods start at 1 not 0
    meta[["t_sim", "t_gt", "T"]] += 1

    # --- 6.  RLP ------------------------------------------------------------- #
    meta["lag"] = meta["t_sim"] - meta["t_gt"]
    meta["rlp"] = 100 * meta["lag"] / meta["t_gt"]

    return meta["rlp"].mean()

if __name__ == "__main__":
    # ADD simulation result file name here
    # ds
    # filename = "gt_evaluation/gt_experiments_20251005_112105.csv"
    filename="gt_evaluation/gt_experiments_20251005_193803.csv"
    df = pd.read_csv(filename)

    if df["trajectory"].dtype == "object" and isinstance(df["trajectory"].iloc[0], str):
        df["trajectory"] = df["trajectory"].apply(ast.literal_eval)
    
    fip = compute_fip(df)
    mean_rlp = compute_mean_rlp(df)

    print(f"FIP        : {fip:.3f}")
    print(f"Mean RLP   : {mean_rlp:.2f}%")