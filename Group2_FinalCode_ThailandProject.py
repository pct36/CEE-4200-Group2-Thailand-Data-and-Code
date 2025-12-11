#!/usr/bin/env python
# coding: utf-8

# In[209]:


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
is_mm3 = ("mm3" in flow_col.lower()) or ("million" in flow_col.lower())
raw = pd.to_numeric(m[flow_col], errors="coerce")
days = pd.Series(pd.DatetimeIndex(idx).days_in_month, index=idx)
Q_in_cms = (raw.values * 1e6) / (days.values * 86400.0) if is_mm3 else raw.values
I = pd.Series(Q_in_cms, index=idx, name="I_cms").astype(float)

#Constants
rho, g, eta = 1000.0, 9.81, 0.90
H = 8.0 #constant net head (m)
Q_TURB_MAX  = 300.0 #power limit (m^3/s)
Q_PASS_MAX  = 1320.0 #gates/spill capacity (m^3/s) 
EF_BASE = 150.0 #baseline EF (m^3/s) 
K = (rho * g * eta * H) / 1e6 #MW per m^3/s
DT_HOURS = 24.0

def EF(month: int, mult: float = 1.0) -> float:
    base = EF_BASE * (1.2 if month in [5,6,7,8,9] else 1.0)
    return mult * base

#Policies
def policy_closed_gates(I: pd.Series) -> pd.DataFrame:
    a = pd.Series(1, index=I.index, name="a")  # turbines ON
    #total managed release cannot exceed pass capacity
    R_total = np.minimum(np.maximum(I.values, [EF(m) for m in I.index.month]), Q_PASS_MAX)
    Q_turb  = np.minimum(R_total, Q_TURB_MAX) #energy only from turbine portion
    return pd.DataFrame({"I_cms": I, "a": a, "R_cms": R_total, "Q_turb_cms": Q_turb})

def policy_seasonal_open(I: pd.Series) -> pd.DataFrame:
    is_wet = I.index.month.isin([5,6,7,8,9])
    a = pd.Series(1, index=I.index, name="a")
    a[is_wet] = 0 #turbines OFF when gates open

    R = pd.Series(index=I.index, dtype=float)
    #Wet: pass inflow (capped by pass capacity)
    R[is_wet]  = np.minimum(I[is_wet].values, Q_PASS_MAX)
    #Dry: meet EF (capped by pass capacity)
    R[~is_wet] = np.minimum(
        np.maximum([EF(m) for m in I[~is_wet].index.month], I[~is_wet].values),
        Q_PASS_MAX
    )

    Q_turb = pd.Series(np.minimum(R, Q_TURB_MAX), index=I.index, name="Q_turb_cms")
    Q_turb[is_wet] = 0.0
    return pd.DataFrame({"I_cms": I, "a": a, "R_cms": R, "Q_turb_cms": Q_turb})

def policy_year_round_ef(I: pd.Series, ef_mult: float = 1.2) -> pd.DataFrame:
    a = pd.Series(1, index=I.index, name="a")
    R_total = np.minimum(
        np.maximum(I.values, [EF(m, mult=ef_mult) for m in I.index.month]),
        Q_PASS_MAX
    )
    Q_turb = np.minimum(R_total, Q_TURB_MAX)
    return pd.DataFrame({"I_cms": I, "a": a, "R_cms": R_total, "Q_turb_cms": Q_turb})

#Simulation and indicators
def simulate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    t = pd.Series(df.index, index=df.index)
    step_hours = (t.shift(-1) - t).dt.total_seconds().div(3600.0)
    step_hours = step_hours.fillna(method="ffill").fillna(24.0)

    #Power & energy 
    P_MW  = K * df["Q_turb_cms"]
    E_MWh = P_MW * step_hours

    EF_req = pd.Series([EF(m) for m in df.index.month], index=df.index, name="EF_cms")

    out = df.copy()
    out["P_MW"]   = P_MW
    out["E_MWh"]  = E_MWh
    out["EF_cms"] = EF_req

    #What reaches river
    out["Q_river"] = out["Q_turb_cms"]
    if "Seasonal" in label:
        wet = out.index.month.isin([5, 6, 7, 8, 9])
        out.loc[wet, "Q_river"] = out.loc[wet, "I_cms"].clip(upper=Q_PASS_MAX)

    return out

def fdc_mae(managed: pd.Series, natural: pd.Series) -> float:
    """Flow Duration Curves mean absolute error"""
    a = pd.to_numeric(managed, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(natural,  errors="coerce").dropna().to_numpy(dtype=float)

    if a.size == 0 or b.size == 0:
        return float("nan")

    a = np.sort(a)
    b = np.sort(b)
    n = min(a.size, b.size)

    #Align to common length from upper tail
    diff = np.abs(a[-n:] - b[-n:])
    return float(diff.mean())


def _off_share(x: pd.Series) -> float:
    """Share of timesteps with turbines OFF (1.0) vs ON (0.0)"""
    return float((x == 0).mean())

def indicators(ts: pd.DataFrame) -> dict:
    #Hydropower
    yearly_E = ts["E_MWh"].groupby(ts.index.year).sum()
    H1 = float(yearly_E.mean())
    monthly_E = ts["E_MWh"].groupby([ts.index.year, ts.index.month]).sum()
    H2 = float(monthly_E.groupby(level=0).min().mean())

    #Environmental
    E1 = float(fdc_mae(ts["Q_river"], ts["I_cms"]))  
    #Pass: outflow approx. equals inflow and turbines are off
    tol = np.maximum(5.0, 0.05 * ts["I_cms"].to_numpy())  #about 5% or 5 m³/s floor
    is_off = (ts["a"].to_numpy() == 0)
    is_passlike = np.abs(ts["R_cms"].to_numpy() - ts["I_cms"].to_numpy()) <= tol
    pass_series = pd.Series(is_passlike & is_off, index=ts.index, dtype=float)
    #Average share per year, then average across years
    E2 = float(pass_series.groupby(ts.index.year).mean().mean())

    return {
        "H1_AvgAnnual_MWhyr": H1,
        "H2_FirmMonthly_MWh": H2,
        "E1_MAE_m3s":        E1,
        "E2_RunOfRiver_OFF_share": E2,
    }

#Run scenarios and indicator table
scenarios = {}
scenarios["ClosedGates"] = simulate(policy_closed_gates(I), "ClosedGates")
scenarios["SeasonalOpen"] = simulate(policy_seasonal_open(I), "SeasonalOpen")
scenarios["YearRoundEF"]  = simulate(policy_year_round_ef(I, ef_mult=1.05), "YearRoundEF")


def build_summary(scenarios_dict):
    """
    Loop through each scenario, compute indicators, and
    collect everything in a single DataFrame.
    """
    rows = []
    names = []

    for name in scenarios_dict:
        ts = scenarios_dict[name]
        ind_dict = indicators(ts)
        rows.append(ind_dict)
        names.append(name)

    summary_df = pd.DataFrame(rows, index=names)
    return summary_df


summary = build_summary(scenarios)

pd.options.display.float_format = "{:,.2f}".format
print("\nPak Mun — Scenario indicators")
print(summary)

summary_path = OUT / "scenario_indicators.csv"
summary.to_csv(summary_path)

for name in scenarios:
    ts = scenarios[name]

    monthly = ts.groupby([ts.index.year, ts.index.month]).agg(
        MonthlyEnergy_MWh=("E_MWh", "sum"),
        OutflowMin_cms=("R_cms", "min"),
        OutflowMax_cms=("R_cms", "max"),
        EFmean_cms=("EF_cms", "mean"),
        OFF_share=("a", _off_share)
    ).reset_index(names=["Year", "Month"])

    monthly.to_csv(OUT / f"{name}_monthly_table.csv", index=False)

print("\nSaved scenario summary table to:", summary_path)
print("Saved monthly tables for each scenario in:", OUT)


def make_tradeoffs(scenarios_dict):
    """
    Build trade-off tables:
    1) FDC MAE vs mean annual energy
    2) FDC MAE vs firm monthly energy
    """
    energy_rows = []
    firm_rows = []

    for name in scenarios_dict:
        ts = scenarios_dict[name]

        fdc_val = fdc_mae(ts["Q_river"], ts["I_cms"])

        yearly_E = ts["E_MWh"].groupby(ts.index.year).sum()
        mean_ann = float(yearly_E.mean())

        monthly_E = ts["E_MWh"].groupby([ts.index.year, ts.index.month]).sum()
        firm_monthly = float(monthly_E.groupby(level=0).min().mean())

        energy_rows.append([name, fdc_val, mean_ann])
        firm_rows.append([name, fdc_val, firm_monthly])

    trade_df = pd.DataFrame(
        energy_rows,
        columns=["scenario", "FDC_MAE_m3s", "MeanAnnual_MWhyr"]
    )
    trade_df2 = pd.DataFrame(
        firm_rows,
        columns=["scenario", "FDC_MAE_m3s", "FirmMonthly_MWhmo"]
    )

    return trade_df, trade_df2


trade, trade2 = make_tradeoffs(scenarios)


def policy_key(name):
    """
    Map scenario names to a smaller set of legend labels.
    """
    n = name.lower()

    if ("env" in n) or ("ef" in n):
        return "EnvFlows"
    elif ("full" in n) or ("closed" in n) or ("close" in n):
        return "FullHydro"
    elif ("season" in n) or ("open" in n):
        return "Seasonal"
    else:
        return "Other"


COLORS = {
    "EnvFlows": "blue",
    "FullHydro": "orange",
    "Seasonal": "green",
    "Other": "gray"
}

#Plot Hydrologic Alteration vs Mean Annual Energy
plt.figure(figsize=(8, 6))

for i in range(len(trade)):
    scen_name = trade.loc[i, "scenario"]
    fdc_val = trade.loc[i, "FDC_MAE_m3s"]
    mean_E = trade.loc[i, "MeanAnnual_MWhyr"]

    key = policy_key(scen_name)
    c = COLORS.get(key, "gray")

    plt.scatter(fdc_val, mean_E, color=c, s=70)
    plt.annotate(scen_name, (fdc_val, mean_E),
                 xytext=(6, 4), textcoords="offset points")

plt.xlabel("FDC MAE (m³/s)")
plt.ylabel("Mean Annual Energy (MWh/y)")
plt.title("Hydrologic Alteration vs Mean Annual Energy")

handles = [
    mlines.Line2D([], [], color=COLORS["EnvFlows"], marker="o",
                  linestyle="None", label="EnvFlows"),
    mlines.Line2D([], [], color=COLORS["FullHydro"], marker="o",
                  linestyle="None", label="FullHydro"),
    mlines.Line2D([], [], color=COLORS["Seasonal"], marker="o",
                  linestyle="None", label="Seasonal")
]

plt.legend(handles=handles, loc="best")
plt.tight_layout()
plt.savefig(OUT / "tradeoff_energy_vs_fdc_mae.png", dpi=150)


#Plot Hydrologic Alteration vs Firm Monthly Energy
plt.figure(figsize=(8, 6))

for i in range(len(trade2)):
    scen_name = trade2.loc[i, "scenario"]
    fdc_val = trade2.loc[i, "FDC_MAE_m3s"]
    firm_E = trade2.loc[i, "FirmMonthly_MWhmo"]

    key = policy_key(scen_name)
    c = COLORS.get(key, "gray")

    plt.scatter(fdc_val, firm_E, color=c, s=70)
    plt.annotate(scen_name, (fdc_val, firm_E),
                 xytext=(6, 4), textcoords="offset points")

plt.xlabel("FDC MAE (m³/s)")
plt.ylabel("Firm Monthly Energy (MWh/month)")
plt.title("Hydrologic Alteration vs Firm Monthly Energy")

handles = [
    mlines.Line2D([], [], color=COLORS["EnvFlows"], marker="o",
                  linestyle="None", label="EnvFlows"),
    mlines.Line2D([], [], color=COLORS["FullHydro"], marker="o",
                  linestyle="None", label="FullHydro"),
    mlines.Line2D([], [], color=COLORS["Seasonal"], marker="o",
                  linestyle="None", label="Seasonal")
]

plt.legend(handles=handles, loc="best")
plt.tight_layout()
plt.savefig(OUT / "tradeoff_firm_vs_fdc_mae.png", dpi=150)


#Plot managed outflow in 2000
daily_csv1 = BASE / "outputs" / "pak_mun_timeseries.csv"
daily_csv2 = BASE / "pak_mun_timeseries.csv"

if daily_csv1.exists():
    daily_csv = daily_csv1
elif daily_csv2.exists():
    daily_csv = daily_csv2
else:
    daily_csv = None

if daily_csv is not None:
    ts_daily = pd.read_csv(daily_csv, parse_dates=["date"])
    ts_daily = ts_daily.rename(columns={"Q_river": "Qr"})
    ts_daily = ts_daily.set_index("date").sort_index()

    year_plot = 2000
    sel = ts_daily.loc[str(year_plot)]

    plt.figure(figsize=(10, 6))

    for scen_name, grp in sel.groupby("scenario"):
        if ("Env" in scen_name) or ("EF" in scen_name):
            label = "Env"
        elif ("Full" in scen_name) or ("Close" in scen_name):
            label = "Full"
        elif ("Season" in scen_name) or ("Open" in scen_name):
            label = "Seasonal"
        else:
            label = scen_name

        color_map = {"Env": "tab:blue",
                     "Full": "tab:orange",
                     "Seasonal": "tab:green"}

        color_val = color_map.get(label, "gray")
        plt.plot(grp.index, grp["Qr"], label=label,
                 color=color_val, linewidth=2)

    plt.title(f"Managed River Outflow in {year_plot}")
    plt.xlabel("Date")
    plt.ylabel("Q_river (m³/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"managed_outflow_{year_plot}.png", dpi=150)


# In[ ]:





# In[ ]:




