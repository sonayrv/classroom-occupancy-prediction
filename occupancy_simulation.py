"""
occupancy_simulation.py

Reproducible stochastic simulation for the hybrid occupancy
prediction framework. All data are synthetic.
Calibrated for a university with 75% mandatory attendance rule.

Run: python occupancy_simulation.py
"""

import numpy as np
import pandas as pd
from itertools import product

np.random.seed(2024)

# CONFIGURATION

N_ROOMS = 3
N_WEEKS = 20  # ~one heating season Oct–Mar
DAYS_PER_WEEK = 5
N_ENROLLED = 40
CO2_AMB = 400.0  # ppm
CO2_GEN = 18.0  # L/h per person
Q_VENT = 180.0  # m³/h ventilation rate
CO2_SENSOR_STD = 50.0  # ppm, MH-Z19B spec ±50
CANCEL_PROB = 0.06  # 6% unplanned cancellations
STAT_WINDOW_WEEKS = 6
CORRECTION_TAU = 4  # steps
CORRECTION_THRESHOLD = 0.20

# Attendance rate priors (mean, std) by class type
# Calibrated for 75% mandatory attendance policy:
# - Students must attend >=75% of sessions per course
# - Most students attend above minimum to have safety margin
# - Labs are near-mandatory for exam admission
# - Variance is reduced because the floor constraint limits skipping
AR_PRIORS = {
    "lecture": (0.82, 0.06),  # effective mean ~78% after multipliers
    "seminar": (0.87, 0.05),  # effective mean ~83%
    "lab": (0.93, 0.03),      # effective mean ~89%
}

# Class type probabilities per slot
CTYPE_PROBS = {"lecture": 0.45, "seminar": 0.35, "lab": 0.20}

# Slot hours available each day
SLOT_HOURS = [9, 10, 11, 13, 14, 15]

# Day-of-week effect on attendance (Mon=0..Fri=4)
DAY_EFFECT = {0: 1.04, 1: 1.01, 2: 1.00, 3: 0.96, 4: 0.90}

# Hour-of-day effect
HOUR_EFFECT = {
    9: 0.94,
    10: 1.00,
    11: 1.02,
    13: 1.01,
    14: 0.96,
    15: 0.91,
}


def week_effect(w):
    """Semester-week multiplier."""
    if w < 3:
        return 1.08
    if w >= N_WEEKS - 3:
        return 0.91
    return 1.00


# 1. GENERATE SYNTHETIC DATASET


def build_dataset():
    rows = []
    ctypes = list(CTYPE_PROBS.keys())
    cprobs = list(CTYPE_PROBS.values())

    for room, week, day in product(
        range(N_ROOMS), range(N_WEEKS), range(DAYS_PER_WEEK)
    ):
        n_slots = np.random.randint(3, len(SLOT_HOURS) + 1)
        hours = sorted(
            np.random.choice(SLOT_HOURS, size=n_slots, replace=False)
        )
        for h in hours:
            ctype = np.random.choice(ctypes, p=cprobs)
            cancelled = np.random.random() < CANCEL_PROB

            # True attendance rate
            mu, sigma = AR_PRIORS[ctype]
            ar_true = np.random.normal(mu, sigma)
            ar_true *= DAY_EFFECT[day]
            ar_true *= HOUR_EFFECT[h]
            ar_true *= week_effect(week)
            ar_true = np.clip(ar_true, 0.25, 1.00)

            n_actual = (
                0 if cancelled else int(round(ar_true * N_ENROLLED))
            )

            rows.append(
                {
                    "room": room,
                    "week": week,
                    "day": day,
                    "hour": h,
                    "class_type": ctype,
                    "n_enrolled": N_ENROLLED,
                    "cancelled": cancelled,
                    "ar_true": ar_true if not cancelled else 0.0,
                    "n_actual": n_actual,
                }
            )

    return pd.DataFrame(rows)


# 2. PREDICTION METHODS


def pred_schedule(df):
    """Method A: schedule only -> always N_enrolled."""
    return df["n_enrolled"].values.astype(float)


def pred_statistical(df):
    """Method B: schedule x sliding-window attendance rate."""
    preds = np.full(len(df), np.nan)

    for i, row in df.iterrows():
        mask = (
            (df.index < i)
            & (df["day"] == row["day"])
            & (df["hour"] == row["hour"])
            & (df["class_type"] == row["class_type"])
            & (df["week"] >= row["week"] - STAT_WINDOW_WEEKS)
            & (df["week"] < row["week"])
            & (~df["cancelled"])
        )
        hist = df.loc[mask]

        if len(hist) >= 3:
            ar = hist["n_actual"].sum() / hist["n_enrolled"].sum()
        else:
            ar = AR_PRIORS[row["class_type"]][0]

        preds[i] = row["n_enrolled"] * ar

    return preds


def co2_from_people(n, noise=True):
    """Forward model: people -> CO2 (ppm)."""
    c = CO2_AMB + (n * CO2_GEN * 1000) / Q_VENT
    if noise:
        c += np.random.normal(0, CO2_SENSOR_STD)
    return max(c, CO2_AMB)


def people_from_co2(c_meas):
    """Inverse model: CO2 -> estimated people."""
    n = (c_meas - CO2_AMB) * Q_VENT / (CO2_GEN * 1000)
    return max(n, 0.0)


def pred_co2_corrected(df, stat_preds):
    """Method C: statistical + CO2 real-time correction."""
    corrected = stat_preds.copy()

    for i, (_, row) in enumerate(df.iterrows()):
        c_meas = co2_from_people(row["n_actual"])
        n_est = people_from_co2(c_meas)

        # Cancelled-class detection
        if c_meas < 450 and stat_preds[i] > 2:
            corrected[i] = 0.0
            continue

        # Threshold correction
        if stat_preds[i] > 0:
            rel_dev = abs(n_est - stat_preds[i]) / stat_preds[i]
            if rel_dev > CORRECTION_THRESHOLD:
                corrected[i] = stat_preds[i] + (n_est - stat_preds[i])

    return corrected


def pred_hybrid(df, co2_preds):
    """Method D: + gradebook bias feedback."""
    hybrid = co2_preds.copy()

    for ctype in AR_PRIORS:
        for day in range(DAYS_PER_WEEK):
            mask = (df["class_type"].values == ctype) & (
                df["day"].values == day
            )
            if mask.sum() < 5:
                continue
            errors = df["n_actual"].values[mask] - co2_preds[mask]
            bias = np.mean(errors)
            if abs(bias) > 0.8:
                hybrid[mask] += bias * 0.25  # conservative correction

    return hybrid


# 3. METRICS


def metrics(y_true, y_pred, label):
    ae = np.abs(y_true - y_pred)
    mae = np.mean(ae)
    nonzero = y_true > 0
    mape = (
        np.mean(ae[nonzero] / y_true[nonzero]) * 100
        if nonzero.any()
        else 0
    )
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {
        "Method": label,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_%": round(mape, 1),
    }


# 4. ENERGY MODEL


def energy_analysis(df, hybrid_preds):
    """Simplified single-zone energy balance."""
    T_SET = 21.0
    T_SETBACK = 16.0
    T_OUT_MEAN = 5.0
    K_ROOM = 0.15  # kW/°C heat loss coefficient
    DT_H = 1.5  # class duration hours
    Q_PERSON = 0.12  # kW metabolic heat

    base_kwh = 0.0
    adapt_kwh = 0.0

    for i, (_, row) in enumerate(df.iterrows()):
        n_act = row["n_actual"]
        n_pred = max(hybrid_preds[i], 0)
        people_heat = n_act * Q_PERSON * DT_H

        # Baseline: always heat to setpoint
        q_b = max(
            K_ROOM * (T_SET - T_OUT_MEAN) * DT_H - people_heat, 0
        )
        base_kwh += q_b

        # Adaptive: setback if predicted empty
        t_target = T_SETBACK if n_pred < 2 else T_SET
        q_a = max(
            K_ROOM * (t_target - T_OUT_MEAN) * DT_H - people_heat, 0
        )
        adapt_kwh += q_a

    return base_kwh, adapt_kwh


# 5. VALIDATION: CHECK 75% RULE COMPLIANCE


def validate_attendance_policy(df):
    """Verify that simulated attendance respects the 75% rule."""
    print("\n--- 75% Mandatory Attendance Policy Validation ---")
    print(
        "Rule: average attendance per class type must be >= 75%\n"
    )

    all_ok = True
    for ctype in AR_PRIORS:
        m = (df["class_type"] == ctype) & (~df["cancelled"])
        if m.sum() == 0:
            continue
        ar = df.loc[m, "n_actual"].sum() / df.loc[m, "n_enrolled"].sum()
        status = "OK" if ar >= 0.75 else "VIOLATION"
        if ar < 0.75:
            all_ok = False
        print(f"  {ctype:8s}: AR = {ar:.3f} ({ar*100:.1f}%)  [{status}]")

    overall_m = ~df["cancelled"]
    overall_ar = (
        df.loc[overall_m, "n_actual"].sum()
        / df.loc[overall_m, "n_enrolled"].sum()
    )
    overall_status = "OK" if overall_ar >= 0.75 else "VIOLATION"
    if overall_ar < 0.75:
        all_ok = False
    print(
        f"  {'OVERALL':8s}: AR = {overall_ar:.3f} "
        f"({overall_ar*100:.1f}%)  [{overall_status}]"
    )

    if all_ok:
        print("\n  All attendance rates comply with 75% rule.")
    else:
        print(
            "\n  WARNING: Some rates violate the 75% rule!"
        )

    return all_ok


# 6. RUN


def main():
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Random seed: 2024")
    print()

    print("Building synthetic dataset...")
    df = build_dataset()
    y = df["n_actual"].values.astype(float)
    N = len(df)

    print(f"  Total class events : {N}")
    print(
        f"  Cancelled          : {df['cancelled'].sum()} "
        f"({df['cancelled'].mean()*100:.1f}%)"
    )
    print(f"  Mean actual occ.   : {y.mean():.1f}")
    print(f"  Std actual occ.    : {y.std():.1f}")

    # --- validate 75% policy ---
    validate_attendance_policy(df)

    # --- predictions ---
    p_sched = pred_schedule(df)
    print("\nComputing statistical predictions (slow)...")
    p_stat = pred_statistical(df)
    p_co2 = pred_co2_corrected(df, p_stat)
    p_hyb = pred_hybrid(df, p_co2)

    # --- metrics ---
    res = [
        metrics(y, p_sched, "A: Schedule only"),
        metrics(y, p_stat, "B: Schedule+Statistical"),
        metrics(y, p_co2, "C: B + CO2 correction"),
        metrics(y, p_hyb, "D: Full hybrid"),
    ]
    res_df = pd.DataFrame(res)
    print("\n" + res_df.to_string(index=False))

    # Relative improvements
    mA = res[0]["MAPE_%"]
    mB = res[1]["MAPE_%"]
    mC = res[2]["MAPE_%"]
    mD = res[3]["MAPE_%"]
    print(f"\nMAPE reduction A->B : {(1 - mB/mA)*100:.1f}%")
    print(f"MAPE reduction B->C : {(1 - mC/mB)*100:.1f}%")
    print(f"MAPE reduction A->D : {(1 - mD/mA)*100:.1f}%")

    # --- energy ---
    base, adapt = energy_analysis(df, p_hyb)
    sav = (1 - adapt / base) * 100
    print(f"\nEnergy baseline  : {base:.0f} kWh")
    print(f"Energy adaptive  : {adapt:.0f} kWh")
    print(f"Savings          : {sav:.1f}%")

    # --- per-type breakdown ---
    print("\n--- Attendance rate by class type (actual) ---")
    for ct in AR_PRIORS:
        m = (df["class_type"] == ct) & (~df["cancelled"])
        ar = df.loc[m, "n_actual"].sum() / df.loc[m, "n_enrolled"].sum()
        print(f"  {ct:8s}: AR = {ar:.3f}")

    # --- per-type MAPE ---
    print("\n--- MAPE by class type (Full hybrid) ---")
    for ct in AR_PRIORS:
        m = df["class_type"].values == ct
        yt = y[m]
        yp = p_hyb[m]
        nz = yt > 0
        if nz.any():
            mape_ct = (
                np.mean(np.abs(yt[nz] - yp[nz]) / yt[nz]) * 100
            )
            print(f"  {ct:8s}: MAPE = {mape_ct:.1f}%")

    return res_df


if __name__ == "__main__":
    main()
