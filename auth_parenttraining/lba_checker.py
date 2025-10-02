# app_main.py
import streamlit as st
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime, date
from calendar import monthrange
import re
import io
from zipfile import ZipFile

# Optional PDF support via wkhtmltopdf + pdfkit
PDFKIT_AVAILABLE = False
try:
    import pdfkit  # pip install pdfkit ; also install wkhtmltopdf system binary
    PDFKIT_AVAILABLE = True
except Exception:
    PDFKIT_AVAILABLE = False

# ⬇️ helper from utils.py (handles "Last, First M" vs "First M Last", ignores middles/suffixes)
from utils import name_key

st.set_page_config(page_title="Authorization Compliance – Supervision & Parent Training", layout="wide")
st.title("📊 Authorization Hours & Compliance (Aloha Report)")

st.markdown(
    """
Upload your **Aloha report** and a single **Authorization + Expectations** file.

**What this checks**
- **Supervision**: 5% of BT hours actually performed (to-date) per auth window
- **Parent Training**: at least one PT session **each month** in the auth window
- Expected supervision (provided per window or computed by weekly × weeks × %)
- LBA filtering (from Auth/Expected file, not from Aloha)

**Matching note:** Names are matched on **(first,last)** only — middles/initials & suffixes are ignored (e.g., *“Rathbone, Sophie”* ↔ *“Rathbone, Sophie R”*).
"""
)

# ------------------ Uploads ------------------
aloha_file = st.file_uploader("Aloha report (CSV or Excel)", type=["csv", "xlsx", "xls"])
auth_expect_file = st.file_uploader(
    "Authorization + Expectations file (one file) — required: Client Name, Auth Start, Auth End; optional: Weekly Service Hours, Expected Supervision Hours, Supervision LBA",
    type=["csv", "xlsx", "xls"]
)

# ------------------ Cached helpers ------------------
@st.cache_data(show_spinner=False)
def read_any(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

@st.cache_data(show_spinner=False)
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

@st.cache_data(show_spinner=False)
def to_excel_bytes(df_map: dict) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
        for name, df in df_map.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    bio.seek(0)
    return bio.getvalue()

# ------------------ Constants & small utils ------------------
PT_PATTERNS  = ["parent", "family", "caregiver", "pt training", "parent training"]
BT_PATTERNS  = ["bt", "direct service", "technician", "rbt"]
SUP_PATTERNS = ["supervision", "supvr", "lba supervision"]

def colnorm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {
        "Appt Date": "Appt. Date",
        "Appt.  Date": "Appt. Date",
        "Billable Hours": "Billing Hours",
        "Hours": "Billing Hours",
        "Client": "Client Name",
        "Client_name": "Client Name",
        "Auth Start Date": "Auth Start",
        "Authorization Start": "Auth Start",
        "Auth End Date": "Auth End",
        "Authorization End": "Auth End",
        "Weekly Hours": "Weekly Service Hours",
        # LBA aliases (auth/expected files)
        "LBA": "Supervision LBA",
        "LBA Name": "Supervision LBA",
        "Supervisor": "Supervision LBA",
        "Supervisor Name": "Supervision LBA",
        "BCBA": "Supervision LBA",
        "BCBA Name": "Supervision LBA",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df

def norm_lba_list(v) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)): return []
    parts = [p.strip() for p in str(v).replace(";", ",").split(",")]
    return [p for p in parts if p]

def lba_key_set(lst: list[str]) -> set[str]:
    return set(s.lower() for s in lst if s)

def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def has_keyword(s: str, patterns) -> bool:
    if not isinstance(s, str): return False
    txt = s.lower()
    return any(p in txt for p in patterns)

def month_iter(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)
    last = pd.Timestamp(year=end.year, month=end.month, day=1)
    while cur <= last:
        yield cur
        cur = (pd.Timestamp(year=cur.year + 1, month=1, day=1)
               if cur.month == 12 else
               pd.Timestamp(year=cur.year, month=cur.month + 1, day=1))

def weeks_in_range(start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = (end - start).days + 1
    return days / 7.0

def week_bounds(any_date: pd.Timestamp):
    any_date = pd.to_datetime(any_date).normalize()
    monday = any_date - pd.Timedelta(days=any_date.weekday())  # 0=Mon
    sunday = monday + pd.Timedelta(days=6)
    return monday, sunday

def month_bounds(any_date: pd.Timestamp):
    any_date = pd.to_datetime(any_date).normalize()
    y, m = any_date.year, any_date.month
    start = pd.Timestamp(year=y, month=m, day=1)
    last_day = monthrange(y, m)[1]
    end = pd.Timestamp(year=y, month=m, day=last_day)
    return start, end

def html_escape(s):
    return (str(s)
            .replace("&","&amp;")
            .replace("<","&lt;")
            .replace(">","&gt;")
            .replace('"',"&quot;"))

# ------------------ Sidebar defaults ------------------
with st.sidebar:
    st.header("Defaults if not provided per-client / per-window")
    default_weekly = st.number_input("Default Weekly Service Hours", min_value=0.0, step=0.5, value=0.0)
    sup_percent    = st.number_input("Supervision % of Service", min_value=0.0, max_value=100.0, step=0.5, value=5.0)
    sup_fraction   = sup_percent / 100.0
    st.caption("Expected supervision (full period) = weekly × weeks in auth × supervision % (only when not provided per window).")

# ------------------ Read Aloha ------------------
if not aloha_file:
    st.info("⬆️ Upload the Aloha report to begin.")
    st.stop()

aloha = colnorm(read_any(aloha_file))
needed = ["Client Name", "Service Name", "Appt. Date", "Billing Hours"]
miss = [c for c in needed if c not in aloha.columns]
if miss:
    st.error(f"Missing required columns in Aloha report: {miss}")
    st.stop()

# Normalize sessions (Aloha)
aloha["Client Name"]   = aloha["Client Name"].astype(str).str.strip()
aloha["Appt. Date"]    = parse_date_series(aloha["Appt. Date"])
aloha["Billing Hours"] = pd.to_numeric(aloha["Billing Hours"], errors="coerce").fillna(0.0)
aloha["is_parent_training"] = aloha["Service Name"].apply(lambda s: has_keyword(s, PT_PATTERNS))
aloha["is_bt_service"]      = aloha["Service Name"].apply(lambda s: has_keyword(s, BT_PATTERNS))
aloha["is_supervision"]     = aloha["Service Name"].apply(lambda s: has_keyword(s, SUP_PATTERNS))
aloha["name_key"] = aloha["Client Name"].apply(name_key)

# Completed filter (optional)
if "Completed" in aloha.columns:
    if st.toggle("Only include sessions where Completed == 'Yes'", value=True):
        aloha = aloha[aloha["Completed"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])].copy()

# Compute min/max BEFORE date filtering (for sensible defaults)
min_d, max_d = aloha["Appt. Date"].min(), aloha["Appt. Date"].max()

# ------------------ Date Tabs (on-page) ------------------
st.subheader("Select Date Range Mode")
date_tab1, date_tab2, date_tab3 = st.tabs(["Weekly (Mon–Sun)", "Monthly", "Custom"])

with date_tab1:
    anchor = st.date_input(
        "Pick any date in the week",
        value=(min_d.date() if pd.notna(min_d) else date.today()),
        key="week_date"
    )
    if isinstance(anchor, tuple):
        anchor = anchor[0]
    s_start, s_end = week_bounds(pd.to_datetime(anchor))
    st.caption(f"Week: {s_start.date()} → {s_end.date()} (Mon–Sun)")

with date_tab2:
    month_anchor = st.date_input(
        "Pick any date in the month",
        value=(min_d.date() if pd.notna(min_d) else date.today()),
        key="month_date"
    )
    if isinstance(month_anchor, tuple):
        month_anchor = month_anchor[0]
    s_start, s_end = month_bounds(pd.to_datetime(month_anchor))
    st.caption(f"Month: {s_start.date()} → {s_end.date()}")

with date_tab3:
    ses_range = st.date_input(
        "Custom range",
        value=(min_d.date() if pd.notna(min_d) else date.today(),
               max_d.date() if pd.notna(max_d) else date.today()),
        key="custom_date"
    )
    if isinstance(ses_range, tuple):
        s_start, s_end = pd.to_datetime(ses_range[0]), pd.to_datetime(ses_range[1])
    else:
        s_start = s_end = pd.to_datetime(ses_range)
    st.caption(f"Custom: {s_start.date()} → {s_end.date()}")

# Apply session date filter
aloha = aloha[(aloha["Appt. Date"] >= s_start) & (aloha["Appt. Date"] <= s_end)].copy()

# ------------------ Auth periods & per-window hours (SINGLE combined file) ------------------
auth_in_aloha = {"Auth Start", "Auth End"}.issubset(aloha.columns)
if auth_in_aloha:
    maybe_cols = ["Client Name","Auth Start","Auth End","Weekly Service Hours","Expected Supervision Hours","Supervision LBA"]
    present = [c for c in maybe_cols if c in aloha.columns]
    auth_df = colnorm(aloha[present].dropna(how="all"))
else:
    if not auth_expect_file:
        st.error("Please upload the **Authorization + Expectations** file (since Aloha does not include auth columns).")
        st.stop()
    tmp = colnorm(read_any(auth_expect_file))
    req = ["Client Name", "Auth Start", "Auth End"]
    miss2 = [c for c in req if c not in tmp.columns]
    if miss2:
        st.error(f"Authorization+Expectations file missing columns: {miss2}")
        st.stop()
    auth_df = tmp.dropna(subset=req).copy()

# Parse/normalize auth fields
auth_df["Client Name"] = auth_df["Client Name"].astype(str).str.strip()
auth_df["Auth Start"]  = parse_date_series(auth_df["Auth Start"])
auth_df["Auth End"]    = parse_date_series(auth_df["Auth End"])
for c in ["Weekly Service Hours", "Expected Supervision Hours"]:
    if c not in auth_df.columns: auth_df[c] = pd.NA
auth_df["Weekly Service Hours"] = pd.to_numeric(auth_df["Weekly Service Hours"], errors="coerce")
auth_df["Expected Supervision Hours"] = pd.to_numeric(auth_df["Expected Supervision Hours"], errors="coerce")
if "Supervision LBA" not in auth_df.columns:
    auth_df["Supervision LBA"] = pd.NA
auth_df["LBA List (window)"] = auth_df["Supervision LBA"].apply(norm_lba_list)
auth_df["name_key"] = auth_df["Client Name"].apply(name_key)

# Per-window expectations map — keyed by (name_key, start, end)
expect_win_map = {
    (r["name_key"], r["Auth Start"], r["Auth End"]): {
        "weekly": (r["Weekly Service Hours"] if pd.notna(r["Weekly Service Hours"]) else None),
        "expected_sup": (r["Expected Supervision Hours"] if pd.notna(r["Expected Supervision Hours"]) else None),
        "lbas": r["LBA List (window)"],
    }
    for _, r in auth_df.iterrows()
}

# Fallback-by-client expectations map — built from SAME file (aggregated by client)
exp = auth_df.copy()
exp["LBA List (expected)"] = exp["Supervision LBA"].apply(norm_lba_list)
# choose the latest row per client by Auth Start (if multiple)
exp_sorted = exp.sort_values(["name_key", "Auth Start", "Auth End"])
last_per_client = exp_sorted.groupby("name_key", as_index=False).tail(1)
expect_map = {
    r["name_key"]: {
        "weekly": (pd.to_numeric(r.get("Weekly Service Hours"), errors="coerce") if pd.notna(r.get("Weekly Service Hours")) else None),
        "expected_sup": (pd.to_numeric(r.get("Expected Supervision Hours"), errors="coerce") if pd.notna(r.get("Expected Supervision Hours")) else None),
        "lbas": r.get("LBA List (expected)", []),
    }
    for _, r in last_per_client.iterrows()
}

# ------------------ LBA filter (from Auth/Expected only) ------------------
def lba_keys_from_dfcol(series):
    if series is None or len(series) == 0: return set()
    return set().union(*[lba_key_set(x) for x in series])

lbas_auth   = lba_keys_from_dfcol(auth_df["LBA List (window)"]) if len(auth_df) else set()
lbas_expect = set()
for v in expect_map.values():
    lbas_expect |= lba_key_set(v.get("lbas", []))
all_lbas_list = sorted({*(x.title() for x in lbas_auth), *(x.title() for x in lbas_expect)})

st.sidebar.header("LBA filter (from Auth/Expected)")
selected_lbas = st.sidebar.multiselect(
    "Supervision LBA(s)",
    options=all_lbas_list,
    default=all_lbas_list if all_lbas_list else [],
    help="Windows are included if their assigned LBA(s) (from Auth or Expected) intersect this list."
)
selected_lbas_keys = set(s.lower() for s in selected_lbas)
if not selected_lbas:
    selected_label = "No LBA Selected"
elif all_lbas_list and len(selected_lbas) == len(all_lbas_list):
    selected_label = "All LBAs"
elif len(selected_lbas) == 1:
    selected_label = selected_lbas[0]
else:
    selected_label = f"{len(selected_lbas)} LBAs"

slug = re.sub(r"[^A-Za-z0-9]+", "_", selected_label).strip("_").lower() or "all_lbas"

# ------------------ Core computation (build summary per auth window) ------------------
rows = []

for _, w in auth_df.sort_values(["Client Name", "Auth Start"]).iterrows():
    client = w["Client Name"]
    client_key = w["name_key"]
    astart, aend = w["Auth Start"], w["Auth End"]
    if pd.isna(astart) or pd.isna(aend):
        continue

    # Resolve LBA(s) (prefer Auth window; else Expected by client_key)
    win_lbas = w.get("LBA List (window)", [])
    if not win_lbas:
        win_lbas = expect_map.get(client_key, {}).get("lbas", [])
    win_lbas_keys = lba_key_set(win_lbas)

    # LBA filter
    if selected_lbas_keys and not (win_lbas_keys & selected_lbas_keys):
        continue

    lba_tag = ", ".join(win_lbas) if win_lbas else "Unassigned"

    # Subset sessions for this window — match on name_key
    sub = aloha[
        (aloha["name_key"] == client_key) &
        (aloha["Appt. Date"] >= astart) &
        (aloha["Appt. Date"] <= aend)
    ]

    # Hours within window
    pt_hours  = sub.loc[sub["is_parent_training"], "Billing Hours"].sum()
    bt_hours  = sub.loc[sub["is_bt_service"], "Billing Hours"].sum()
    sup_hours = sub.loc[sub["is_supervision"], "Billing Hours"].sum()

    # PT monthly check
    months = list(month_iter(astart, aend))
    today = pd.Timestamp.today().normalize()
    current_month = pd.Timestamp(year=today.year, month=today.month, day=1)
    months_to_check = [m for m in months if m <= current_month]

    pt_months = set(pd.to_datetime(sub.loc[sub["is_parent_training"], "Appt. Date"]).dt.strftime("%Y-%m").tolist())
    missing_months = [m.strftime("%Y-%m") for m in months_to_check if m.strftime("%Y-%m") not in pt_months]
    pt_ok = (len(missing_months) == 0)

    # Expected supervision (full period)
    wks = weeks_in_range(astart, aend)
    win_key = (client_key, astart, aend)
    weekly = expect_win_map.get(win_key, {}).get("weekly", None)
    if weekly is None:
        weekly = expect_map.get(client_key, {}).get("weekly", None)
    if weekly is None:
        weekly = default_weekly

    explicit_expected = expect_win_map.get(win_key, {}).get("expected_sup", None)
    if explicit_expected is not None:
        expected_sup_full = float(explicit_expected)
        expected_basis = "provided"
    else:
        expected_sup_full = (weekly or 0.0) * wks * sup_fraction
        expected_basis = "computed"

    # 5% target based on BT hours actually performed in this window
    sup_needed_from_bt_done = bt_hours * sup_fraction
    delta_vs_bt5 = sup_hours - sup_needed_from_bt_done
    sup5_ok = delta_vs_bt5 >= 0

    # Full-period “max” supervision based on weekly plan (always computed)
    max_sup_full = (weekly or 0.0) * wks * sup_fraction

    # Expected BT service hours for the full auth period (plan × weeks)
    expected_bt_full = (weekly or 0.0) * wks

    rows.append({
        # Identity & window
        "Client Name": client,
        "Auth Start": astart.date(),
        "Auth End": aend.date(),
        "Weeks in Auth": round(wks, 2),
        "Supervision LBA": lba_tag,

        # Planning / expectations
        "Planned Weekly Service Hours": round(float(weekly or 0.0), 2),
        "Expected BT Service Hours (auth period)": round(float(expected_bt_full), 2),
        "Max Supervision Hours (plan × weeks × 5%)": round(float(max_sup_full), 2),
        "Expectation Source": expected_basis,

        # Actuals to date
        "BT Hours (Done)": round(float(bt_hours), 2),
        "Supervision Hours (Done)": round(float(sup_hours), 2),
        "Parent Training Hours (Done)": round(float(pt_hours), 2),

        # Supervision compliance (against BT actually done)
        "Supervision 5% of BT (to date)": round(float(sup_needed_from_bt_done), 2),
        "Δ Supervision (done - 5% of BT done)": round(float(delta_vs_bt5), 2),
        "Supervision 5% OK": bool(sup5_ok),
        "Supervision Shortfall (hrs)": round(float(max(0.0, sup_needed_from_bt_done - sup_hours)), 2),

        # PT compliance
        "PT Monthly OK": bool(pt_ok),
        "PT Missing Months": ", ".join(missing_months) if missing_months else "",
    })

# Build summary dataframe
summary = pd.DataFrame(rows).sort_values(["Client Name", "Auth Start"]).reset_index(drop=True)

# ------------------ Upcoming Reassessments (filtered by LBA) ------------------
today = pd.Timestamp.today().normalize()
lookahead_days = 45
lookahead = today + pd.Timedelta(days=lookahead_days)

reassess_rows = []
for _, r in auth_df.iterrows():
    client = r["Client Name"]; client_key = r["name_key"]
    astart, aend = r["Auth Start"], r["Auth End"]
    if pd.isna(aend): continue

    # Resolve LBA(s) & filter
    win_lbas = r.get("LBA List (window)", []) or expect_map.get(client_key, {}).get("lbas", [])
    win_lbas_keys = lba_key_set(win_lbas)
    if selected_lbas_keys and not (win_lbas_keys & selected_lbas_keys):
        continue
    lba_tag = ", ".join(win_lbas) if win_lbas else "Unassigned"

    # Due rule: using Auth End
    due = aend
    if due < today:
        status = "Overdue"
    elif due <= lookahead:
        status = "Upcoming"
    else:
        status = "Not Due Yet"

    reassess_rows.append({
        "Client Name": client,
        "Auth Start": astart.date() if pd.notna(astart) else None,
        "Auth End": aend.date() if pd.notna(aend) else None,
        "Reassessment Due": due.date(),
        "Status": status,
        "Supervision LBA": lba_tag
    })

reassess_df = pd.DataFrame(reassess_rows).sort_values("Reassessment Due")
if not reassess_df.empty and "Status" in reassess_df.columns:
    status_map = {"Overdue":"🔴 Overdue", "Upcoming":"🟡 Upcoming", "Not Due Yet":"🟢 Not Due Yet"}
    reassess_df["Status"] = reassess_df["Status"].map(lambda s: status_map.get(s, s))

# ------------------ PT missing-month details table ------------------
pt_missing_detail_rows = []
if not summary.empty and "PT Missing Months" in summary.columns:
    for _, r in summary.iterrows():
        if r.get("PT Missing Months"):
            for m in str(r["PT Missing Months"]).split(", "):
                if m.strip():
                    pt_missing_detail_rows.append({
                        "Client Name": r["Client Name"],
                        "Auth Start": r["Auth Start"],
                        "Auth End": r["Auth End"],
                        "Missing Month": m.strip(),
                        "Supervision LBA": r.get("Supervision LBA", "Unassigned")
                    })
pt_missing_detail = pd.DataFrame(pt_missing_detail_rows)

# Parent-Training-focused view
pt_cols = [
    "Client Name","Auth Start","Auth End","Supervision LBA",
    "Parent Training Hours (Done)","PT Monthly OK","PT Missing Months"
]
pt_view = summary[pt_cols].copy() if not summary.empty else pd.DataFrame(columns=pt_cols)
pt_hours_cols = ["Client Name","Auth Start","Auth End","Supervision LBA","Parent Training Hours (Done)"]
pt_hours_view = summary[pt_hours_cols].copy() if not summary.empty else pd.DataFrame(columns=pt_hours_cols)

# ------------------ One-page HTML (Summary + Reassessments; no Totals) ------------------
def table_from_df(dfx: pd.DataFrame, row_classes=None) -> str:
    cols = list(dfx.columns)
    thead = "<thead><tr>" + "".join(f"<th>{html_escape(c)}</th>" for c in cols) + "</tr></thead>"
    rows_html = []
    for i, row in dfx.iterrows():
        cls = ""
        if row_classes is not None and i < len(row_classes):
            rc = row_classes[i]
            if isinstance(rc, bool):
                rc = "bad" if rc else "ok"
            elif rc not in ("ok", "bad"):
                rc = "ok"
            cls = f" class='{rc}'"
        tds = "".join(f"<td>{html_escape(v)}</td>" for v in row.values)
        rows_html.append(f"<tr{cls}>{tds}</tr>")
    return f"<table>{thead}<tbody>{''.join(rows_html)}</tbody></table>"

def one_page_html_colored(
    summary: pd.DataFrame,
    reassess_df: pd.DataFrame,
    pt_df: pd.DataFrame | None = None,
    sup_percent: float = 5.0,
    title_text: str = "Authorization Compliance – One-Page Report",
) -> str:
    def to_bool(x):
        s = str(x).strip().lower()
        if s in ("true","yes","1","✓","ok"): return True
        if s in ("false","no","0","✗","x"): return False
        return None

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------- Supervision summary (no PT columns here) ----------
    sup_pref = [
        "Client Name","Auth Start","Auth End","Supervision LBA",
        "BT Hours (Done)","Supervision Hours (Done)",
        "Supervision 5% of BT (to date)","Δ Supervision (done - 5% of BT done)",
        "Supervision 5% OK",
    ]
    sup_cols_present = [c for c in sup_pref if c in summary.columns]
    sup_df = summary[sup_cols_present].copy()

    # Derive 5% OK defensively if missing
    if "Supervision 5% OK" not in sup_df.columns and {"BT Hours (Done)","Supervision Hours (Done)"}.issubset(sup_df.columns):
        bt = pd.to_numeric(sup_df["BT Hours (Done)"], errors="coerce").fillna(0.0)
        sup = pd.to_numeric(sup_df["Supervision Hours (Done)"], errors="coerce").fillna(0.0)
        sup_df["Supervision 5% of BT (to date)"] = (bt * (sup_percent/100.0)).round(2)
        sup_df["Supervision 5% OK"] = (sup >= sup_df["Supervision 5% of BT (to date)"])

    for c in ["Supervision 5% OK"]:
        if c in sup_df.columns:
            sup_df[c] = sup_df[c].map(lambda x: "✓" if to_bool(x) is True else ("✗" if to_bool(x) is False else x))

    # Keep page compact
    max_rows = 35
    if len(sup_df) > max_rows:
        sup_df = sup_df.head(max_rows).reset_index(drop=True)

    # Row coloring: bad if 5% not met
    sup_row_classes = []
    if "Supervision 5% OK" in sup_df.columns:
        sup_row_classes = [("bad" if (to_bool(v) is False) else "ok") for v in sup_df["Supervision 5% OK"]]
    else:
        sup_row_classes = ["ok"] * len(sup_df)

    # ---------- Parent Training table (separate section) ----------
    pt_table_html = "<div style='font-size:10px;color:#666;'>No parent-training data for this window.</div>"
    if pt_df is not None and not pt_df.empty:
        pt_pref = [
            "Client Name","Auth Start","Auth End","Supervision LBA",
            "Parent Training Hours (Done)","PT Monthly OK","PT Missing Months"
        ]
        pt_cols_present = [c for c in pt_pref if c in pt_df.columns]
        pt_tbl = pt_df[pt_cols_present].copy()

        # Iconify booleans
        if "PT Monthly OK" in pt_tbl.columns:
            pt_tbl["PT Monthly OK"] = pt_tbl["PT Monthly OK"].map(
                lambda x: "✓" if to_bool(x) is True else ("✗" if to_bool(x) is False else x)
            )

        # Limit rows
        if len(pt_tbl) > max_rows:
            pt_tbl = pt_tbl.head(max_rows).reset_index(drop=True)

        pt_table_html = table_from_df(pt_tbl)

    style = """
    <style>
      @page { size: Letter; margin: 0.5in; }
      body { font-family: Arial, Helvetica, sans-serif; margin: 0.5in; color:#111; }
      h1 { font-size: 20px; margin: 0 0 6px 0; }
      .meta { color: #555; font-size: 12px; margin-bottom: 8px; }
      .section { margin-top: 12px; }
      table { width: 100%; border-collapse: collapse; font-size: 10px; }
      th, td { border: 1px solid #ccc; padding: 4px 6px; }
      th { background: #f5f5f5; text-align: left; }
      tr.bad { background: #ffeaea; }
      tr.ok  { background: #f2fbf2; }
    </style>
    """

    reassess_html = (reassess_df.to_html(index=False)
                     if reassess_df is not None and not reassess_df.empty
                     else "<div style='font-size:10px;color:#666;'>No reassessments for the selected LBA(s) in this window.</div>")

    html = f"""<!doctype html><html><head><meta charset='utf-8'>{style}
    <title>{html_escape(title_text)}</title></head>
    <body>
      <h1>{html_escape(title_text)}</h1>
      <div class="meta">Generated: {now}</div>

      <div class="section"><strong>Supervision Compliance</strong>{table_from_df(sup_df, sup_row_classes)}</div>

      <div class="section"><strong>Parent Training Compliance</strong>{pt_table_html}</div>

      <div class="section"><strong>Upcoming Reassessments</strong>{reassess_html}</div>
    </body></html>"""
    return html

# ------------------ Per-LBA helpers ------------------
def filtered_views_for_lba(lba_name: str, summary_df: pd.DataFrame, pt_df: pd.DataFrame, reassess: pd.DataFrame):
    # Build a robust mask for "contains" across possibly multi-LBA cells (e.g., "Alice, Bob")
    if lba_name and lba_name.lower() != "unassigned":
        mask_sum = summary_df["Supervision LBA"].astype(str).str.lower().str.contains(lba_name.lower(), regex=False)
        mask_pt  = pt_df["Supervision LBA"].astype(str).str.lower().str.contains(lba_name.lower(), regex=False) if pt_df is not None and not pt_df.empty else None
        mask_re  = reassess["Supervision LBA"].astype(str).str.lower().str.contains(lba_name.lower(), regex=False) if reassess is not None and not reassess.empty else None
    else:
        # Unassigned bucket
        mask_sum = summary_df["Supervision LBA"].fillna("").str.strip().eq("") | summary_df["Supervision LBA"].fillna("Unassigned").eq("Unassigned")
        mask_pt  = (pt_df["Supervision LBA"].fillna("").str.strip().eq("") | pt_df["Supervision LBA"].eq("Unassigned")) if pt_df is not None and not pt_df.empty else None
        mask_re  = (reassess["Supervision LBA"].fillna("").str.strip().eq("") | reassess["Supervision LBA"].eq("Unassigned")) if reassess is not None and not reassess.empty else None

    sum_lba = summary_df[mask_sum].reset_index(drop=True)
    pt_lba = (pt_df[mask_pt].reset_index(drop=True) if (pt_df is not None and not pt_df.empty and mask_pt is not None) else pt_df)
    re_lba = (reassess[mask_re].reset_index(drop=True) if (reassess is not None and not reassess.empty and mask_re is not None) else reassess)
    return sum_lba, pt_lba, re_lba

def make_per_lba_zip(all_lbas: list[str], summary_df: pd.DataFrame, pt_df: pd.DataFrame, reassess: pd.DataFrame,
                     sup_percent: float, window_label: str, as_pdf: bool = False) -> bytes:
    # If no LBAs detected, produce one “All LBAs” report using current filter result.
    targets = all_lbas[:] if all_lbas else ["All LBAs"]

    buf = io.BytesIO()
    with ZipFile(buf, "w") as zf:
        for lba in targets:
            # For "All LBAs", don't filter anything—use the whole current summary/pt/reassess
            if lba == "All LBAs":
                sum_lba = summary_df
                pt_lba  = pt_df
                re_lba  = reassess
            else:
                sum_lba, pt_lba, re_lba = filtered_views_for_lba(lba, summary_df, pt_df, reassess)

            if (sum_lba is None or sum_lba.empty) and (pt_lba is None or pt_lba.empty) and (re_lba is None or re_lba.empty):
                # skip empty reports
                continue

            title = f"Authorization Compliance – {lba} ({window_label})"
            html = one_page_html_colored(sum_lba, re_lba, pt_df=pt_lba, sup_percent=sup_percent, title_text=title)

            safe_lba = re.sub(r"[^A-Za-z0-9]+", "_", lba).strip("_").lower() or "all_lbas"
            if as_pdf and PDFKIT_AVAILABLE:
                try:
                    pdf_bytes = pdfkit.from_string(html, False)  # returns bytes
                    zf.writestr(f"{safe_lba}.pdf", pdf_bytes)
                except Exception:
                    # Fallback to HTML if conversion fails
                    zf.writestr(f"{safe_lba}.html", html.encode("utf-8"))
            else:
                zf.writestr(f"{safe_lba}.html", html.encode("utf-8"))

    buf.seek(0)
    return buf.getvalue()

# ------------------ Build report HTML for current (filtered) view ------------------
window_label = f"{pd.to_datetime(s_start).date()} to {pd.to_datetime(s_end).date()}"
report_title = f"Authorization Compliance – {selected_label} ({window_label})"
html_report = one_page_html_colored(
    summary,
    reassess_df,
    pt_df=pt_view,
    sup_percent=sup_percent,
    title_text=report_title
)

# ------------------ Split views for tabs ------------------
# Supervision-focused view
sup_cols = [
    "Client Name","Auth Start","Auth End","Supervision LBA",
    "BT Hours (Done)","Supervision Hours (Done)",
    "Supervision 5% of BT (to date)",
    "Δ Supervision (done - 5% of BT done)",
    "Supervision Shortfall (hrs)",
    "Planned Weekly Service Hours",
    "Expected BT Service Hours (auth period)",
    "Max Supervision Hours (plan × weeks × 5%)",
    "Expectation Source",
    "Supervision 5% OK",
]
sup_view = summary[sup_cols].copy() if not summary.empty else pd.DataFrame(columns=sup_cols)

# ------------------ Main content tabs ------------------
tab_sup, tab_pt, tab_re, tab_exp = st.tabs(["Supervision", "Parent Training", "Reassessments", "One-Pager & Exports"])

with tab_sup:
    st.subheader("Supervision Compliance (5% of BT)")
    if sup_view.empty:
        st.info("No results — check filters / date range.")
    else:
        sup5_noncomp = (~summary["Supervision 5% OK"]).sum()
        if sup5_noncomp:
            st.error(f"❗ {sup5_noncomp} auth window(s) are below the required {sup_percent:.1f}% supervision based on BT hours performed.")
        else:
            st.success(f"✅ Supervision meets or exceeds {sup_percent:.1f}% for all windows.")
        # Sort to bubble up shortfalls
        sup_view_sorted = sup_view.sort_values(
            ["Supervision 5% OK", "Supervision Shortfall (hrs)"],
            ascending=[True, False]
        )
        st.dataframe(sup_view_sorted, use_container_width=True)

with tab_pt:
    st.subheader("Parent Training Compliance (Monthly)")
    if pt_view.empty:
        st.info("No results — check filters / date range.")
    else:
        pt_noncomp = (~summary["PT Monthly OK"]).sum()
        if pt_noncomp:
            st.error(f"❗ {pt_noncomp} auth window(s) are missing Parent Training month(s).")
        else:
            st.success("✅ Parent Training monthly requirement satisfied for all windows.")
        st.dataframe(pt_view, use_container_width=True)

        with st.expander("Show details for missing Parent Training months"):
            if pt_missing_detail.empty:
                st.write("No missing months.")
            else:
                st.dataframe(pt_missing_detail, use_container_width=True)

with tab_re:
    st.subheader("Upcoming Reassessments")
    if reassess_df.empty:
        st.info("No reassessments due for the selected LBA(s) in the chosen date window.")
    else:
        st.caption("Status meanings: Overdue 🔴 • Upcoming 🟡 • Not Due Yet 🟢")
        st.dataframe(reassess_df, use_container_width=True)

with tab_exp:
    st.subheader("Download One-Pager (HTML) & Data Exports")

    st.download_button(
        "🖨️ Download One-Page Report (HTML)",
        data=html_report.encode("utf-8"),
        file_name=f"auth_one_page_report_{slug}.html",
        mime="text/html"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ Download Supervision Summary (CSV)",
            data=to_csv_bytes(sup_view),
            file_name=f"supervision_summary_{slug}.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "⬇️ Download Parent Training Summary (CSV)",
            data=to_csv_bytes(pt_view),
            file_name=f"parent_training_summary_{slug}.csv",
            mime="text/csv"
        )
    with c3:
        xbytes = to_excel_bytes({
            "Supervision": sup_view,
            "Parent Training": pt_view,
            "Auth Windows Used": auth_df.sort_values(["Client Name", "Auth Start", "Auth End"]).reset_index(drop=True),
            "Reassessments": (reassess_df if not reassess_df.empty else pd.DataFrame(columns=["Client Name","Auth Start","Auth End","Reassessment Due","Status","Supervision LBA"]))
        })
        st.download_button(
            "⬇️ Download Combined Results (Excel)",
            data=xbytes,
            file_name=f"auth_results_{slug}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown("---")
    st.subheader("Per-LBA One-Pager (bulk export)")

    if not all_lbas_list:
        st.info("No LBAs detected from the Authorization/Expectations data.")
    else:
        zl_html = make_per_lba_zip(all_lbas_list, summary, pt_view, reassess_df, sup_percent, window_label, as_pdf=False)
        st.download_button(
            "📦 Download Per-LBA One-Pagers (ZIP • HTML files)",
            data=zl_html,
            file_name=f"per_lba_onepagers_html_{slug}.zip",
            mime="application/zip"
        )

        if PDFKIT_AVAILABLE:
            zl_pdf = make_per_lba_zip(all_lbas_list, summary, pt_view, reassess_df, sup_percent, window_label, as_pdf=True)
            st.download_button(
                "📦 Download Per-LBA One-Pagers (ZIP • PDF files)",
                data=zl_pdf,
                file_name=f"per_lba_onepagers_pdf_{slug}.zip",
                mime="application/zip"
            )
        else:
            st.caption("PDF export requires `wkhtmltopdf` + `pdfkit`. Install the wkhtmltopdf binary and `pip install pdfkit`, then restart the app.")
