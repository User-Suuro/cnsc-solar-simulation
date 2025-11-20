import matplotlib

matplotlib.use("Agg")

from flask import Flask, render_template, request, send_file, redirect, url_for, session, abort
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

app.secret_key = "replace_this_with_a_real_secret"

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CONS_XLSX = DATA_DIR / "Consumption.xlsx"
IRR_CSV = DATA_DIR / "Monthlydata_14.248_122.767_E5_2014_2023.csv"
OUT_DIR = DATA_DIR / "output_sim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "panel_kw": 284.0,
    "battery_kwh": 1334.0,
    "inverter_kw": 85.0,
}


# ----------------- UTILITIES -----------------

def safe_read_irr_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError("Missing irradiation CSV")
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "\nyear\t"
    pos = text.find(marker)
    if pos == -1:
        pos = text.find("year\t")
    if pos == -1:
        raise ValueError("Cannot find 'year' header in irradiation file")
    csv_part = text[pos + 1:] if text[pos] == "\n" else text[pos:]
    df = pd.read_csv(io.StringIO(csv_part), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    hcol = None
    for c in df.columns:
        lc = c.lower()
        if "h(h)" in lc or "h_h" in lc or "irradiation" in lc:
            hcol = c
            break
    if hcol is None:
        for c in reversed(df.columns):
            try:
                _ = df[c].astype(float)
                hcol = c
                break
            except Exception:
                continue
    if hcol is None:
        raise ValueError("Cannot find irradiation column in CSV")

    df["year"] = df["year"].astype(str).str.strip()
    df["month"] = df["month"].astype(str).str.strip().str[:3]
    return df, hcol


def read_simple_consumption(path: Path):
    if not path.exists():
        raise FileNotFoundError("Missing consumption Excel")
    df = pd.read_excel(path, sheet_name=0, header=0, index_col=0)
    df.index = df.index.astype(str).str.strip()
    new_cols = []
    for c in df.columns:
        try:
            new_cols.append(int(c))
        except Exception:
            try:
                new_cols.append(int(str(c).strip()))
            except Exception:
                new_cols.append(c)
    df.columns = new_cols
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def ensure_12_months(lst):
    arr = [0.0 if pd.isna(x) else float(x) for x in list(lst)]
    if len(arr) >= 12:
        return arr[:12]
    else:
        avg = float(np.mean([v for v in arr if not pd.isna(v)])) if arr else 0.0
        pad = [avg] * (12 - len(arr))
        return arr + pad


# ---------- UPDATED: Linear Regression for future consumption ----------
def get_yearly_consumption(cons_df: pd.DataFrame, year: int):
    hist_years = sorted([c for c in cons_df.columns if isinstance(c, int)])
    if not hist_years:
        return [0.0] * 12

    last_hist = hist_years[-1]

    if year in hist_years:
        monthly = cons_df[year].tolist()

    elif year > last_hist:
        X = np.array(hist_years).reshape(-1, 1)
        monthly = []
        for month_idx in range(12):
            y_vals = []
            for y in hist_years:
                try:
                    y_vals.append(float(cons_df[y].iloc[month_idx]))
                except Exception:
                    y_vals.append(0.0)
            y_vals = np.array(y_vals)
            model = LinearRegression()
            model.fit(X, y_vals)
            pred = model.predict(np.array([[year]]))[0]
            monthly.append(float(pred))
    else:
        monthly = cons_df[hist_years[0]].tolist()

    return ensure_12_months(monthly)


def get_monthly_irr_for_year(df_irr: pd.DataFrame, hcol: str, year: int):
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    if df_irr is None or df_irr.empty:
        return [4.7] * 12

    hist_years = sorted(int(y) for y in df_irr["year"].unique() if y.isdigit())
    last_hist = hist_years[-1] if hist_years else 2023

    if int(year) <= last_hist:
        dfy = df_irr[df_irr["year"] == str(year)]
    else:
        dfy = pd.DataFrame()
        monthly_avg = {}
        for m in order:
            vals = []
            for y in hist_years:
                try:
                    v = float(df_irr.loc[(df_irr["year"] == str(y)) & (df_irr["month"] == m), hcol].values[0])
                    vals.append(v)
                except Exception:
                    continue
            monthly_avg[m] = float(np.mean(vals)) if vals else 4.7
        hm = [monthly_avg[m] for m in order]
        return ensure_12_months(hm)

    hm = []
    available_vals = []
    for m in order:
        try:
            v = float(dfy.loc[dfy["month"] == m, hcol].values[0])
        except Exception:
            v = np.nan
        available_vals.append(v)

    av_nonan = [v for v in available_vals if not pd.isna(v)]
    pad = float(np.mean(av_nonan)) if av_nonan else 4.7
    for v in available_vals:
        hm.append(float(pad if pd.isna(v) else v))

    return ensure_12_months(hm)


# ------------------- UPDATED simulate_annual() WITH MONTHLY DEGRADATION -------------------
def simulate_annual(panel_kw, battery_kwh, inverter_kw, monthly_irr, monthly_con, initial_health=100.0,
                    return_deg=False):
    system_eff = 0.8
    battery_soc_max = battery_kwh * 0.8
    soc = battery_soc_max
    charge_eff = 0.9
    discharge_eff = 0.9
    health = float(initial_health)

    # Monthly degradation = 0.8% / 12 = 0.0666667%
    monthly_deg = 0.8 / 12.0

    monthly_irr = ensure_12_months(monthly_irr)
    monthly_con = ensure_12_months(monthly_con)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    rows = []
    deg_rows = []  # For degradation analysis

    for i, m in enumerate(months):
        Hm = float(monthly_irr[i])

        # Calculate solar generation at 100% health (for comparison)
        solar_gen_100pct = panel_kw * Hm * system_eff

        # Calculate actual solar generation with current health
        panel_eff_kw = float(panel_kw) * (health / 100.0)
        solar_gen_actual = panel_eff_kw * Hm * system_eff

        # Calculate degradation loss
        degradation_loss = solar_gen_100pct - solar_gen_actual

        consumption = float(monthly_con[i])
        net = solar_gen_actual - consumption
        met = 0.0
        unmet = 0.0

        if net >= 0:
            charge = min(net * charge_eff, battery_soc_max - soc)
            soc += charge
            met = consumption
        else:
            deficit = -net
            draw = min(deficit / discharge_eff, soc)
            soc -= draw
            remaining = deficit - draw * discharge_eff
            if remaining > 1e-9:
                unmet = remaining
                met = solar_gen_actual + draw * discharge_eff
            else:
                met = consumption

        rows.append({
            "Month": m,
            "SolarGen_kWh": round(solar_gen_actual, 2),
            "Consumption_kWh": round(consumption, 2),
            "Met_kWh": round(met, 2),
            "Unmet_kWh": round(unmet, 2),
            "BatterySOC_end_kWh": round(soc, 2),
            "Health_after_month": round(health, 3),
        })

        # Add to degradation analysis table
        deg_rows.append({
            "Month": m,
            "SolarGen_100pct_kWh": round(solar_gen_100pct, 2),
            "SolarGen_actual_kWh": round(solar_gen_actual, 2),
            "DegradationLoss_kWh": round(degradation_loss, 2),
            "Health_pct": round(health, 2),
        })

        # Apply monthly degradation
        health -= monthly_deg

    df = pd.DataFrame(rows)
    df_deg = pd.DataFrame(deg_rows)  # Degradation analysis dataframe

    summary = {
        "TotalSolar_kWh": round(float(df["SolarGen_kWh"].sum()), 2),
        "TotalConsumption_kWh": round(float(df["Consumption_kWh"].sum()), 2),
        "TotalMet_kWh": round(float(df["Met_kWh"].sum()), 2),
        "TotalUnmet_kWh": round(float(df["Unmet_kWh"].sum()), 2),
        "FinalBatterySOC_kWh": round(float(df["BatterySOC_end_kWh"].iloc[-1]), 2),
        "FinalHealth": round(float(df["Health_after_month"].iloc[-1]), 2),
        # Add degradation summary
        "TotalSolar_100pct_kWh": round(float(df_deg["SolarGen_100pct_kWh"].sum()), 2),
        "TotalDegradationLoss_kWh": round(float(df_deg["DegradationLoss_kWh"].sum()), 2),
        "InitialHealth": round(initial_health, 2),  # Track initial health for multi-year
    }

    if return_deg:
        return df, df_deg, summary, health  # Return final health for next year
    else:
        return df, df_deg, summary


# ----------------- ROUTES -----------------

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        df_irr, hcol = safe_read_irr_csv(IRR_CSV)
    except Exception as e:
        print("Irradiation load error:", e)
        df_irr = pd.DataFrame()
        hcol = None

    try:
        cons_df = read_simple_consumption(CONS_XLSX)
    except Exception as e:
        print("Consumption load error:", e)
        cons_df = pd.DataFrame(
            index=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        cons_df[2024] = 0.0

    all_years = list(range(2017, 2038))
    panel_kw = float(DEFAULTS["panel_kw"])
    battery_kwh = float(DEFAULTS["battery_kwh"])
    inverter_kw = float(DEFAULTS["inverter_kw"])
    year_start = year_end = 2025

    # Initialize all output variables
    single_monthly_table = single_summary = single_deg_table = None
    bar_url = pie_url = excel_url = None
    multi_summary_table = multi_deg_table = None
    multi_line_url = multi_excel_url = None
    multi_bar_url = multi_pie_url = multi_deg_trend_url = None

    if request.method == "POST":
        try:
            panel_kw = float(request.form.get("panel_kw", panel_kw))
            battery_kwh = float(request.form.get("battery_kwh", battery_kwh))
            inverter_kw = float(request.form.get("inverter_kw", inverter_kw))
            year_start = int(request.form.get("year_start", year_start))
            year_end = int(request.form.get("year_end", year_end))
        except Exception:
            pass

        year_start = max(2017, min(2037, year_start))
        year_end = max(year_start, min(2037, year_end))

        # ---------------- SINGLE YEAR ----------------
        if year_start == year_end:
            year = year_start
            monthly_irr = get_monthly_irr_for_year(df_irr, hcol, year)
            monthly_con = get_yearly_consumption(cons_df, year)

            # CHANGED: Now receives 3 return values
            df_monthly, df_deg, summary = simulate_annual(
                panel_kw, battery_kwh, inverter_kw, monthly_irr, monthly_con
            )

            bar_url = f"bar_{year}.png"
            pie_url = f"pie_{year}.png"
            excel_url = f"sim_{year}.xlsx"

            # bar plot
            fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
            x = np.arange(len(df_monthly))
            ax.bar(x - 0.2, df_monthly["SolarGen_kWh"], width=0.4, label="SolarGen (kWh)")
            ax.bar(x + 0.2, df_monthly["Consumption_kWh"], width=0.4, label="Consumption (kWh)")
            ax.set_xticks(x)
            ax.set_xticklabels(df_monthly["Month"])
            ax.set_ylabel("Energy (kWh)")
            ax.set_title(f"Monthly Solar vs Consumption ({year})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT_DIR / bar_url, bbox_inches="tight")
            plt.close(fig)

            # pie plot
            met_val = summary.get("TotalMet_kWh", 0.0)
            unmet_val = summary.get("TotalUnmet_kWh", 0.0)
            pie_vals = [float(met_val), float(unmet_val)]
            pie_labels = ["Met", "Unmet"] if sum(pie_vals) > 0 else ["No data"]
            if sum(pie_vals) == 0:
                pie_vals = [1.0]
            figp, axp = plt.subplots(figsize=(6, 6), dpi=100)
            axp.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%")
            axp.set_title("Annual Met vs Unmet Energy")
            figp.tight_layout()
            figp.savefig(OUT_DIR / pie_url, bbox_inches="tight")
            plt.close(figp)

            # Excel
            with pd.ExcelWriter(OUT_DIR / excel_url) as writer:
                df_monthly.to_excel(writer, index=False, sheet_name="Monthly")
                df_deg.to_excel(writer, index=False, sheet_name="Degradation Analysis")
                pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="Summary")

            single_monthly_table = df_monthly.round(2).to_dict(orient="records")
            single_deg_table = df_deg.round(2).to_dict(orient="records")
            single_summary = summary

        # ---------------- MULTI-YEAR ----------------
        else:
            summary_rows = []
            deg_summary_rows = []  # For multi-year degradation summary
            health = 100.0

            for y in range(year_start, year_end + 1):
                monthly_irr = get_monthly_irr_for_year(df_irr, hcol, y)
                monthly_con = get_yearly_consumption(cons_df, y)

                # Get both summary and degradation data for multi-year
                _, df_deg_year, summary, final_health = simulate_annual(
                    panel_kw, battery_kwh, inverter_kw,
                    monthly_irr, monthly_con,
                    initial_health=health,
                    return_deg=True  # Request degradation data
                )

                # Add to yearly summary
                summary_row = {"Year": y, **summary}
                summary_rows.append(summary_row)

                # Calculate yearly degradation summary
                yearly_deg_summary = {
                    "Year": y,
                    "Initial_Health_%": round(health, 2),
                    "Final_Health_%": round(final_health, 2),
                    "Health_Loss_%": round(health - final_health, 2),
                    "Solar_At_100%_kWh": round(float(df_deg_year["SolarGen_100pct_kWh"].sum()), 2),
                    "Solar_Actual_kWh": round(float(df_deg_year["SolarGen_actual_kWh"].sum()), 2),
                    "Degradation_Loss_kWh": round(float(df_deg_year["DegradationLoss_kWh"].sum()), 2),
                    "Degradation_Loss_%": round((float(df_deg_year["DegradationLoss_kWh"].sum()) /
                                                 float(df_deg_year["SolarGen_100pct_kWh"].sum()) * 100)
                                                if float(df_deg_year["SolarGen_100pct_kWh"].sum()) > 0 else 0, 2)
                }
                deg_summary_rows.append(yearly_deg_summary)

                # Update health for next year (use the final health from this year)
                health = final_health

            # Create main summary table
            df_summary = pd.DataFrame(summary_rows)
            cols = df_summary.columns.tolist()
            if "Year" in cols:
                cols = ["Year"] + [c for c in cols if c != "Year"]
                df_summary = df_summary[cols]

            # Create degradation summary table
            df_deg_summary = pd.DataFrame(deg_summary_rows)

            multi_summary_table = df_summary.round(2).to_dict(orient="records")
            multi_deg_table = df_deg_summary.round(2).to_dict(orient="records")

            # Line chart
            multi_line_url = f"multi_line_{year_start}_{year_end}.png"
            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
            ax.plot(df_summary["Year"], df_summary["TotalConsumption_kWh"], marker="o", label="Total Consumption")
            ax.plot(df_summary["Year"], df_summary["TotalMet_kWh"], marker="o", label="Total Met")
            ax.plot(df_summary["Year"], df_summary["TotalUnmet_kWh"], marker="o", label="Total Unmet")
            ax.plot(df_summary["Year"], df_summary["TotalSolar_kWh"], marker="o", label="Total Solar Generated")
            ax.set_xlabel("Year")
            ax.set_ylabel("Energy (kWh)")
            ax.set_title(f"Yearly Energy Summary ({year_start}-{year_end})")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            fig.tight_layout()
            fig.savefig(OUT_DIR / multi_line_url, bbox_inches="tight")
            plt.close(fig)

            # Degradation trend chart
            multi_deg_trend_url = f"multi_deg_trend_{year_start}_{year_end}.png"
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=100)

            # Health trend
            ax1.plot(df_deg_summary["Year"], df_deg_summary["Initial_Health_%"], marker="o", label="Health %",
                     color="red")
            ax1.set_ylabel("Health (%)")
            ax1.set_title(f"System Health Degradation Trend ({year_start}-{year_end})")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.5)

            # Degradation loss trend
            ax2.bar(df_deg_summary["Year"], df_deg_summary["Degradation_Loss_kWh"], color="orange", alpha=0.7,
                    label="Degradation Loss")
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Energy Loss (kWh)")
            ax2.set_title("Annual Degradation Energy Loss")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.5)

            fig.tight_layout()
            fig.savefig(OUT_DIR / multi_deg_trend_url, bbox_inches="tight")
            plt.close(fig)

            # Bar chart
            multi_bar_url = f"multi_bar_{year_start}_{year_end}.png"
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            x = np.arange(len(df_summary))
            ax.bar(x - 0.2, df_summary["TotalSolar_kWh"], width=0.4, label="SolarGen (kWh)")
            ax.bar(x + 0.2, df_summary["TotalConsumption_kWh"], width=0.4, label="Consumption (kWh)")
            ax.set_xticks(x)
            ax.set_xticklabels(df_summary["Year"])
            ax.set_ylabel("Energy (kWh)")
            ax.set_title(f"Yearly Solar vs Consumption ({year_start}-{year_end})")
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUT_DIR / multi_bar_url, bbox_inches="tight")
            plt.close(fig)

            # Pie chart
            total_met = df_summary["TotalMet_kWh"].sum()
            total_unmet = df_summary["TotalUnmet_kWh"].sum()
            pie_vals = [total_met, total_unmet]
            pie_labels = ["Met", "Unmet"] if sum(pie_vals) > 0 else ["No data"]
            if sum(pie_vals) == 0:
                pie_vals = [1.0]
            multi_pie_url = f"multi_pie_{year_start}_{year_end}.png"
            figp, axp = plt.subplots(figsize=(6, 6), dpi=100)
            axp.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%")
            axp.set_title("Total Met vs Unmet Energy (All Years)")
            figp.tight_layout()
            figp.savefig(OUT_DIR / multi_pie_url, bbox_inches="tight")
            plt.close(figp)

            # Excel - UPDATED to include degradation sheet
            multi_excel_url = f"multi_{year_start}_{year_end}.xlsx"
            with pd.ExcelWriter(OUT_DIR / multi_excel_url) as writer:
                df_summary.to_excel(writer, index=False, sheet_name="Yearly Summary")
                df_deg_summary.to_excel(writer, index=False, sheet_name="Degradation Summary")

    return render_template(
        "index.html",
        years=all_years,
        year_start=year_start,
        year_end=year_end,
        panel_kw=panel_kw,
        battery_kwh=battery_kwh,
        inverter_kw=inverter_kw,
        health=round(session.get("health", 100.0), 2),
        single_monthly_table=single_monthly_table,
        single_deg_table=single_deg_table,
        single_summary=single_summary,
        bar_url=bar_url,
        pie_url=pie_url,
        excel_url=excel_url,
        multi_summary_table=multi_summary_table,
        multi_deg_table=multi_deg_table,
        multi_line_url=multi_line_url,
        multi_deg_trend_url=multi_deg_trend_url,
        multi_bar_url=multi_bar_url,
        multi_pie_url=multi_pie_url,
        multi_excel_url=multi_excel_url
    )


@app.route("/output/<filename>")
def serve_output(filename):
    path = OUT_DIR / filename
    if not path.exists():
        abort(404)
    return send_file(str(path))


@app.route("/download/<filename>")
def download_file(filename):
    path = OUT_DIR / filename
    if not path.exists():
        return redirect(url_for("index"))
    return send_file(str(path), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)