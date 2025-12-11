#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#Paths
BASE = Path.home() / "Desktop" / "Data" / "PakMun"
INFLOW = BASE / "Pak_Mun_Dam_Inflow.csv" 
OUT    = BASE / "outputs"
OUT.mkdir(exist_ok=True, parents=True)

#Load monthly inflow and build monthly index
m = pd.read_csv(INFLOW, low_memory=False)

cols_lower = {c.lower(): c for c in m.columns}
if "year" in cols_lower and "month" in cols_lower:
    yy = pd.to_numeric(m[cols_lower["year"]], errors="coerce").astype("Int64")
    mm = pd.to_numeric(m[cols_lower["month"]], errors="coerce").astype("Int64")
    idx = pd.to_datetime(
        pd.DataFrame({"year": yy, "month": mm, "day": 1}),
        errors="coerce"
    )

candidates = [c for c in m.columns if any(k in c.lower() for k in ["mm3", "qinflow", "q_in", "inflow", "flow"])]
stats = []
for c in candidates:
    s = pd.to_numeric(m[c], errors="coerce")
    stats.append((c, s.notna().sum(), float(np.nanvar(s))))
flow_col = sorted(stats, key=lambda t: (-t[1], -t[2]))[0][0]

#Convert to m^3/s 
raw = pd.to_numeric(m[flow_col], errors="coerce")
days = pd.Series(pd.DatetimeIndex(idx).days_in_month, index=idx)
Q_in_cms = (raw.values * 1e6) / (days.values * 86400.0) if np.nanmax(raw.values) > 1e4 else raw.values
I = pd.Series(Q_in_cms, index=idx, name="I_cms").astype(float)

#Constants
rho, g, eta = 1000.0, 9.81, 0.90
H = 8.0 #constant net head (m)
Q_TURB_MAX = 300.0 #turbine capacity (m^3/s)
EF_BASE = 150.0 #baseline EF (m^3/s) 
K = (rho * g * eta * H) / 1e6 #MW per m^3/s
DT_HOURS = 24.0

def EF(month: int, mult: float = 1.0) -> float:
    base = EF_BASE * (1.2 if month in [5,6,7,8,9] else 1.0)
    return mult * base

#Policies
def policy_closed_gates(I: pd.Series) -> pd.DataFrame:
    """Hydro ON every month. Release meets EF, turbine limited"""
    a = pd.Series(1, index=I.index, name="a")  # turbines ON
    R = pd.Series(np.maximum(I.values, [EF(m) for m in I.index.month]), index=I.index, name="R_cms")
    Q_turb = np.minimum(R, Q_TURB_MAX)
    return pd.DataFrame({"I_cms": I, "a": a, "R_cms": R, "Q_turb_cms": Q_turb})

def policy_seasonal_open(I: pd.Series) -> pd.DataFrame:
    """Gates open (turbines OFF) in wet months: pass inflow. Dry months use turbines up to capacity with EF floor"""
    is_wet = I.index.month.isin([5,6,7,8,9])
    a = pd.Series(1, index=I.index, name="a")
    a[is_wet] = 0
    R = pd.Series(index=I.index, dtype=float)
    R[is_wet]  = I[is_wet].values
    R[~is_wet] = np.maximum([EF(m) for m in I[~is_wet].index.month],
                            np.minimum(I[~is_wet].values, Q_TURB_MAX))
    Q_turb = pd.Series(np.minimum(R, Q_TURB_MAX), index=I.index, name="Q_turb_cms")
    Q_turb[is_wet] = 0.0
    return pd.DataFrame({"I_cms": I, "a": a, "R_cms": R, "Q_turb_cms": Q_turb})

def policy_year_round_ef(I: pd.Series, ef_mult: float = 1.2) -> pd.DataFrame:
    """Hydro ON year-round. EF every month"""
    a = pd.Series(1, index=I.index, name="a")
    R = pd.Series(np.maximum(I.values, [EF(m, mult=ef_mult) for m in I.index.month]), index=I.index, name="R_cms")
    Q_turb = np.minimum(R, Q_TURB_MAX)
    return pd.DataFrame({"I_cms": I, "a": a, "R_cms": R, "Q_turb_cms": Q_turb})

#Simulation and indicators
def simulate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute power/energy and discharge reaching the river (Q_river)"""
    P_MW  = K * df["Q_turb_cms"]
    E_MWh = P_MW * DT_HOURS
    EF_req = pd.Series([EF(m) for m in df.index.month], index=df.index, name="EF_cms")

    out = df.copy()
    out["P_MW"]   = P_MW
    out["E_MWh"]  = E_MWh
    out["EF_cms"] = EF_req

    if "Seasonal" in label:
        wet = out.index.month.isin([5,6,7,8,9])
        out["Q_river"] = out["Q_turb_cms"]
        out.loc[wet, "Q_river"] = out.loc[wet, "I_cms"]
    elif "ClosedGates" in label:
        out["Q_river"] = out["Q_turb_cms"]
    else:  # YearRoundEF
        out["Q_river"] = out["Q_turb_cms"]

    return out

def fdc_mae(a: pd.Series, b: pd.Series) -> float:
    """Mean absolute difference between FDCs"""
    aa = np.sort(np.asarray(a.dropna(), dtype=float))
    bb = np.sort(np.asarray(b.dropna(), dtype=float))
    n = min(len(aa), len(bb))
    return float(np.mean(np.abs(aa[-n:] - bb[-n:]))) if n else np.nan

def indicators(ts: pd.DataFrame) -> dict:
    #Hydropower
    yearly_E   = ts["E_MWh"].groupby(ts.index.year).sum()
    H1         = yearly_E.mean()
    monthly_E  = ts["E_MWh"].groupby([ts.index.year, ts.index.month]).sum()
    H2         = monthly_E.groupby(level=0).min().mean()

    #Environmental
    E1 = fdc_mae(ts["Q_river"], ts["I_cms"])  
    eps = np.maximum(5.0, 0.05 * ts["I_cms"].values)
    pass_like = (np.abs(ts["R_cms"].values - ts["I_cms"].values) <= eps) & (ts["a"].values == 0)
    E2 = pd.Series(pass_like.astype(float), index=ts.index).groupby(ts.index.year).mean().mean()

    return {"H1_AvgAnnual_MWhyr": float(H1),
            "H2_FirmMonthly_MWh": float(H2),
            "E1_MAE_m3s":        float(E1),
            "E2_RunOfRiver_OFF_share": float(E2)}

#Run scenarios and indicator table
scenarios = {
    "ClosedGates":  simulate(policy_closed_gates(I),                "ClosedGates"),
    "SeasonalOpen": simulate(policy_seasonal_open(I),               "SeasonalOpen"),
    "YearRoundEF":  simulate(policy_year_round_ef(I, ef_mult=1.05),  "YearRoundEF"),
}

rows = [pd.Series(indicators(ts), name=name) for name, ts in scenarios.items()]
summary = pd.DataFrame(rows)
pd.options.display.float_format = "{:,.2f}".format
print("\nPak Mun — Scenario indicators")
print(summary)

summary.to_csv(OUT / "scenario_indicators.csv")

# per-scenario monthly summary tables
for name, ts in scenarios.items():
    monthly = ts.groupby([ts.index.year, ts.index.month]).agg(
        MonthlyEnergy_MWh=("E_MWh","sum"),
        OutflowMin_cms=("R_cms","min"),
        OutflowMax_cms=("R_cms","max"),
        EFmean_cms=("EF_cms","mean"),
        OFF_share=("a", lambda x: (x==0).mean())
    )
    monthly.to_csv(OUT / f"{name}_monthly_table.csv")
print(f"\nSaved: {OUT/'scenario_indicators.csv'} and one monthly_table per scenario in {OUT}")

#Trade-off tables 
rows1, rows2 = [], []
for name, ts in scenarios.items():
    fdc  = fdc = fdc_mae(ts["Q_river"], ts["I_cms"])
    Ey   = ts["E_MWh"].groupby(ts.index.year).sum()
    mean_ann = float(Ey.mean())
    Em   = ts["E_MWh"].groupby([ts.index.year, ts.index.month]).sum()
    firm = float(Em.groupby(level=0).min().mean())
    rows1.append((name, fdc, mean_ann))
    rows2.append((name, fdc, firm))

trade  = pd.DataFrame(rows1, columns=["scenario","FDC_MAE_m3s","MeanAnnual_MWhyr"])
trade2 = pd.DataFrame(rows2, columns=["scenario","FDC_MAE_m3s","FirmMonthly_MWhmo"])

#Plots
def policy_key(name: str) -> str:
    n = name.lower()
    if "env" in n or "ef" in n:
        return "EnvFlows"
    if "full" in n or "closed" in n or "close" in n:
        return "FullHydro"
    if "season" in n or "open" in n:
        return "Seasonal"
    return "Other"

COLORS = {"EnvFlows": "blue", "FullHydro": "orange", "Seasonal": "green", "Other": "gray"}

#Plot Alteration vs Mean Annual Energy
plt.figure(figsize=(8,6))
for _, r in trade.iterrows():
    key = policy_key(r["scenario"]); c = COLORS[key]
    plt.scatter(r["FDC_MAE_m3s"], r["MeanAnnual_MWhyr"], color=c, s=70)
    plt.annotate(r["scenario"], (r["FDC_MAE_m3s"], r["MeanAnnual_MWhyr"]),
                 xytext=(6,4), textcoords="offset points")
plt.xlabel("FDC MAE (m³/s) — lower is better")
plt.ylabel("Mean Annual Energy (MWh/y) — higher is better")
plt.title("Hydrologic Alteration vs Mean Annual Energy")
handles = [mlines.Line2D([], [], color=COLORS[k], marker='o', linestyle='None', label=k)
           for k in ["EnvFlows", "FullHydro", "Seasonal"]]
plt.legend(handles=handles, loc="best")
plt.tight_layout()
plt.savefig(OUT / "tradeoff_energy_vs_fdc_mae.png", dpi=150)

#Plot Alteration vs Firm Monthly Energy
plt.figure(figsize=(8,6))
for _, r in trade2.iterrows():
    key = policy_key(r["scenario"]); c = COLORS[key]
    plt.scatter(r["FDC_MAE_m3s"], r["FirmMonthly_MWhmo"], color=c, s=70)
    plt.annotate(r["scenario"], (r["FDC_MAE_m3s"], r["FirmMonthly_MWhmo"]),
                 xytext=(6,4), textcoords="offset points")
plt.xlabel("FDC MAE (m³/s) — lower is better")
plt.ylabel("Firm Monthly Energy (MWh/month) — higher is better")
plt.title("Hydrologic Alteration vs Firm Monthly Energy")
handles = [mlines.Line2D([], [], color=COLORS[k], marker='o', linestyle='None', label=k)
           for k in ["EnvFlows", "FullHydro", "Seasonal"]]
plt.legend(handles=handles, loc="best")
plt.tight_layout()
plt.savefig(OUT / "tradeoff_firm_vs_fdc_mae.png", dpi=150)

#Managed outflow in 2000 
daily_csv = (BASE / "outputs" / "pak_mun_timeseries.csv"
             if (BASE / "outputs" / "pak_mun_timeseries.csv").exists()
             else BASE / "pak_mun_timeseries.csv")
if daily_csv.exists():
    ts_daily = (pd.read_csv(daily_csv, parse_dates=["date"])
                  .rename(columns={"Q_river":"Qr"})
                  .set_index("date")
                  .sort_index())
    year = 2000
    sel = ts_daily.loc[str(year)]

    plt.figure(figsize=(10,6))
    for name, g in sel.groupby("scenario"):
        label = ("Env" if "Env" in name or "EF" in name else
                 "Full" if "Full" in name or "Close" in name else
                 "Seasonal" if "Season" in name or "Open" in name else name)
        color = {"Env":"tab:blue","Full":"tab:orange","Seasonal":"tab:green"}.get(label, "gray")
        plt.plot(g.index, g["Qr"], label=label, color=color, linewidth=2)

    plt.title(f"Managed River Outflow in {year}")
    plt.xlabel("Date"); plt.ylabel("Q_river (m³/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"managed_outflow_{year}.png", dpi=150)

