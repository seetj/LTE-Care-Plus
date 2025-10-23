# combined_checker_minimal.py
import os
import re
import math
import time
from difflib import SequenceMatcher

import pandas as pd
import requests
import streamlit as st

# =============== App Config ===============
st.set_page_config(page_title="AlohaABA — Match + Duration + Geolocation", layout="wide")
st.title("AlohaABA — Match + Duration + Geolocation (No Aggregation)")

# ------------- Fixed config (no user choices) -------------
MAPBOX_TOKEN = (
    os.getenv("MAPBOX_ACCESS_TOKEN")
    or st.secrets.get("MAPBOX_ACCESS_TOKEN", None)
)
DISTANCE_FEET_THRESHOLD = 800
DISTANCE_METERS_THRESHOLD = DISTANCE_FEET_THRESHOLD * 0.3048
DEFAULT_STATE_CLEAN = "NY"  # always NY
SESSION_REQUIRED = "1:1 BT Direct Service"
STATUS_REQUIRED = "Transferred to AlohaABA"

# Sessions required columns
COL_CLIENT_ADDR = "Client: Address Line 1"
COL_CLIENT_CITY = "Client: City / District"
COL_CLIENT_ZIP  = "Client: Zip / Postal Code"
SESS_REQUIRED_COLS = [
    "Client", "User", "Date/Time", "End time", "Duration", "Session", "Status",
    "User signature", "User signature location",
    "Parent signature", "Parent signature location",
    COL_CLIENT_ADDR, COL_CLIENT_CITY, COL_CLIENT_ZIP
]

# Billing required columns (predefined)
BILL_REQUIRED_COLS = ["Staff Name", "Client Name","Appt. Date", "Billing Hours"]

# Output columns
END_OUTPUT_COLS = [
    "Client", "BT", "Date/Time", "End time", "Duration", "Session", "Status",
    COL_CLIENT_ADDR, COL_CLIENT_CITY, COL_CLIENT_ZIP,
    "Client lat,lon",
    "User signature location", "User signature address",
    "Parent signature location", "Parent signature address",
    "Billing matched staff", "Billing matched client", "Billing date",
    "Billing hours (min)", "Session duration (min)", "Duration diff (min)", "Duration OK?",
    "Matched staff score", "Matched client score",
    "Geo OK?", "Issue Type",
    "Reason",
]

# =============== Helpers ===============
def read_any(f):
    if f.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(f)
        except UnicodeDecodeError:
            f.seek(0); return pd.read_csv(f, encoding="latin-1")
    return pd.read_excel(f)

try:
    from unidecode import unidecode
except Exception:
    unidecode = lambda s: s

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def norm_name(s: str) -> str:
    if s is None: return ""
    s = unidecode(str(s).strip())
    if "," in s:  # "Last, First, Suffix"
        parts = [p.strip() for p in s.split(",") if p is not None]
        last = parts[0] if len(parts) >= 1 else ""
        first_middle = parts[1] if len(parts) >= 2 else ""
        suffix = parts[2] if len(parts) >= 3 else ""
        s = f"{first_middle} {last} {suffix}".strip()
    s = re.sub(r"[^A-Za-z\s]", " ", s)
    toks = re.sub(r"\s+", " ", s).strip().lower().split()
    toks = [t for t in toks if (len(t) > 1) and (t not in SUFFIXES)]
    return " ".join(toks)

def last_name(raw: str) -> str:
    n = norm_name(raw); parts = n.split()
    return parts[-1] if parts else ""

def token_set_score(a_raw: str, b_raw: str) -> float:
    a = norm_name(a_raw); b = norm_name(b_raw)
    if not a or not b: return 0.0
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb: return 0.0
    overlap = len(ta & tb)
    return (2.0 * overlap) / (len(ta) + len(tb))

def name_similarity(a_raw: str, b_raw: str) -> float:
    base = token_set_score(a_raw, b_raw)
    la, lb = last_name(a_raw), last_name(b_raw)
    if la and lb and la == lb:
        base = min(1.0, base + 0.15)
    return base

def clean_addr(addr: str) -> str:
    if not addr: return ""
    return re.sub(r"\s+", " ", str(addr)).strip()

ZIP_CLEAN_RE = re.compile(r"[^0-9-]")
UNIT_TOKEN_RE = re.compile(r"\b(apt|apartment|unit|#|suite|ste|fl|floor|room|rm)\b", re.I)

def clean_zip(z) -> str:
    if z is None or (isinstance(z, float) and math.isnan(z)): return ""
    s = str(z).strip()
    if s.endswith(".0"): s = s[:-2]
    s = ZIP_CLEAN_RE.sub("", s)
    if re.fullmatch(r"\d{4}", s): s = "0" + s
    if re.fullmatch(r"\d{9}", s): s = s[:5] + "-" + s[5:]
    m = re.match(r"^(\d{5})(?:-(\d{4}))?$", s)
    return f"{m.group(1)}-{m.group(2)}" if (m and m.group(2)) else (m.group(1) if m else "")

def split_primary_unit(addr: str):
    s = clean_addr(addr)
    if not s: return "", ""
    m = UNIT_TOKEN_RE.search(s)
    if not m: return s, ""
    return s[:m.start()].rstrip(", "), s[m.start():].lstrip(", ").strip()

def parse_latlon(s: str):
    if s is None or pd.isna(s): return None
    s = str(s).strip().replace("(", "").replace(")", "")
    if not s or "," not in s: return None
    parts = s.split(",")
    if len(parts) != 2: return None
    try:
        lat = float(parts[0].strip()); lon = float(parts[1].strip())
        if -90 <= lat <= 90 and -180 <= lon <= 180: return (lat, lon)
    except Exception:
        return None
    return None

def haversine_m(a, b):
    if not a or not b: return None
    R = 6371000.0
    lat1, lon1 = a; lat2, lon2 = b
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    x = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1-x))

def parse_hhmm_to_minutes(v) -> float:
    if v is None: return 0.0
    s = str(v).strip()
    if not s: return 0.0
    if ":" not in s:
        try: return float(s)
        except Exception: return 0.0
    parts = s.split(":")
    try:
        h = int(parts[0]); m = int(parts[1]) if len(parts) > 1 else 0
        s_ = int(parts[2]) if len(parts) > 2 else 0
        return float(h * 60 + m) + (s_ / 60.0)
    except Exception:
        return 0.0

def detect_bill_units(series) -> str:
    """
    Auto-detect billing units:
    - If any value > 12 and no value with ':' -> assume Minutes
    - Else assume Decimal hours
    """
    s = pd.to_numeric(pd.to_datetime(series, errors="coerce"), errors="coerce")  # bait datetime to NaT; then numeric
    # If dates got converted, treat separately
    as_str = series.astype(str)
    if (as_str.str.contains(":").any()):
        # someone accidentally put HH:MM — treat as minutes via parse_hhmm_to_minutes
        return "HHMM"
    vals = pd.to_numeric(series.astype(str).str.strip().str.replace(",", ""), errors="coerce")
    if (vals > 12).any():
        return "MIN"
    return "HRS"

def to_minutes_session(val) -> float:
    # Sessions Duration assumed HH:MM or HH:MM:SS (your exports)
    return parse_hhmm_to_minutes(val)

def to_minutes_billing(val, detected: str) -> float:
    if detected == "HHMM":
        return parse_hhmm_to_minutes(val)
    if detected == "MIN":
        try: return float(str(val).strip())
        except Exception: return 0.0
    # "HRS"
    try: return float(str(val).strip()) * 60.0
    except Exception: return 0.0

# Mapbox helpers
def _mapbox_forward_detail(q: str, token: str, timeout=5):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(q)}.json"
    params = {"access_token": token, "limit": 1, "country": "US", "types": "address", "autocomplete": "false"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        js = r.json(); feats = js.get("features", [])
        if feats:
            f0 = feats[0]; lon, lat = f0.get("center", [None, None])
            meta = {
                "text": f0.get("text", ""), "place_name": f0.get("place_name", ""),
                "matching_text": f0.get("matching_text", ""), "relevance": f0.get("relevance", None),
            }
            if lat is not None and lon is not None:
                return (lat, lon, meta)
    except Exception:
        pass
    return (None, None, {})

def _mapbox_reverse(lat, lon, token: str, timeout=5):
    if lat is None or lon is None: return ""
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {"access_token": token, "limit": 1, "country": "US"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        js = r.json(); feats = js.get("features", [])
        if feats: return feats[0].get("place_name", "") or ""
    except Exception:
        return ""
    return ""

def uniq_nonempty(parts):
    seen = set(); out = []
    for p in parts:
        p = (p or "").strip().strip(",")
        if not p or p in seen: continue
        seen.add(p); out.append(p)
    return out

def forward_candidates(addr, city, state, zip_):
    primary, unit = split_primary_unit(addr)
    city = (city or "").strip(); state = (state or "").strip().upper()[:2]; zip_ = clean_zip(zip_)
    base = uniq_nonempty([primary, city, state, zip_])
    with_unit = uniq_nonempty([f"{primary}, {unit}" if unit else primary, city, state, zip_])
    candidates = [", ".join(with_unit), ", ".join(base), ", ".join(base[:-1]), f"{primary} {zip_}".strip(), primary]
    return uniq_nonempty(candidates)

def robust_forward(addr, city, state, zip_, token, retries=1):
    tried = set()
    for q in forward_candidates(addr, city, state, zip_):
        q = q.strip().strip(",")
        if not q or q in tried: continue
        tried.add(q)
        lat, lon, meta = _mapbox_forward_detail(q, token)
        if lat is not None and lon is not None:
            return (lat, lon, f"OK: {q}", q, meta)
        if retries:
            lat, lon, meta = _mapbox_forward_detail(q, token)
            if lat is not None and lon is not None:
                return (lat, lon, f"OK(retry): {q}", q, meta)
    tried_list = list(tried)
    return (None, None, f"NO_MATCH after tries: {', '.join(tried_list[:3])}…", "", {})

# =============== Uploads ===============
left, right = st.columns(2)
with left:
    sess_file = st.file_uploader("Upload Sessions file (CSV/XLSX)", type=["csv", "xlsx", "xls"])
with right:
    bill_file = st.file_uploader("Upload Billing file (CSV/XLSX)", type=["csv", "xlsx", "xls"])

if not sess_file:
    st.info("Upload a Sessions file to begin.")
    st.stop()

# =============== Sessions load & filter ===============
sessions = read_any(sess_file)
missing_s = [c for c in SESS_REQUIRED_COLS if c not in sessions.columns]
if missing_s:
    st.error(f"Sessions: missing columns {missing_s}")
    st.stop()

st.caption("Preview: Sessions (first 20 rows)")
st.dataframe(sessions.head(20), use_container_width=True)

# Filter to required session + status
work = sessions[
    (sessions["Session"].astype(str).str.strip() == SESSION_REQUIRED)
    & (sessions["Status"].astype(str).str.strip() == STATUS_REQUIRED)
].copy()

if work.empty:
    st.warning("No rows meet the filters: Session='1:1 BT Direct Service' AND Status='Transferred to AlohaABA'.")
    st.stop()

work.rename(columns={"User": "BT"}, inplace=True)
work.reset_index(drop=True, inplace=True)
work["Session date"] = pd.to_datetime(work["Date/Time"], errors="coerce").dt.date
work[COL_CLIENT_CITY] = work[COL_CLIENT_CITY].astype(str).str.strip()
work[COL_CLIENT_ZIP] = work[COL_CLIENT_ZIP].apply(clean_zip)

# Initialize outputs
work["Reason"] = ""
work["Client lat,lon"] = ""
work["User signature address"] = ""
work["Parent signature address"] = ""
work["Session duration (min)"] = work["Duration"].apply(to_minutes_session)
work["Duration diff (min)"] = None
work["Duration OK?"] = None
work["Geo OK?"] = None
work["Billing matched staff"] = ""
work["Billing matched client"] = ""
work["Billing date"] = ""
work["Billing hours (min)"] = None
work["Matched staff score"] = None
work["Matched client score"] = None

# Parse signatures + immediate flags
user_coords, parent_coords = [], []
pre_reasons = []
for _, row in work.iterrows():
    r = []
    u_sig = str(row.get("User signature", "")).strip()
    p_sig = str(row.get("Parent signature", "")).strip()
    u_ll = parse_latlon(row.get("User signature location"))
    p_ll = parse_latlon(row.get("Parent signature location"))

    if not u_sig: r.append("No user signature")
    elif u_ll is None: r.append("User signature present but no location")
    if not p_sig: r.append("No parent signature")
    elif p_ll is None: r.append("Parent signature present but no location")

    user_coords.append(u_ll); parent_coords.append(p_ll)
    pre_reasons.append(", ".join(sorted(set(r))))
work["Reason"] = pre_reasons

# =============== Matching + Duration (no aggregation) ===============
if not bill_file:
    st.warning("Upload a Billing file to run matching and duration checks.")
else:
    billing = read_any(bill_file)
    missing_b = [c for c in BILL_REQUIRED_COLS if c not in billing.columns]
    if missing_b:
        st.error(f"Billing: missing columns {missing_b}")
        st.stop()

    st.caption("Preview: Billing (first 20 rows)")
    st.dataframe(billing.head(20), use_container_width=True)

    bill = billing.copy()
    bill["_bill_date"] = pd.to_datetime(bill["Appt. Date"], errors="coerce").dt.date
    bill["_norm_staff"] = bill["Staff Name"].astype(str).map(norm_name)
    bill["_norm_client"] = bill["Client Name"].astype(str).map(norm_name)
    # auto-detect units for Billing Hours
    detected_units = detect_bill_units(bill["Billing Hours"])
    bill["_bill_minutes"] = bill["Billing Hours"].apply(lambda v: to_minutes_billing(v, detected_units))
    bill["_used"] = False

    work["_norm_staff"] = work["BT"].astype(str).map(norm_name)
    work["_norm_client"] = work["Client"].astype(str).map(norm_name)

    STAFF_T = 0.85
    CLIENT_T = 0.85
    TOL_MIN = 7  # fixed tolerance

    for idx, row in work.iterrows():
        s_date = row["Session date"]
        s_staff_raw = row.get("BT", "")
        s_client_raw = row.get("Client", "")

        cand = bill[bill["_bill_date"] == s_date].copy()
        if cand.empty:
            work.at[idx, "Reason"] = (row["Reason"] + ", No billing line match (date+staff+client)").strip(", ")
            work.at[idx, "Duration OK?"] = False
            continue

        cand["_staff_score"]  = cand["Staff Name"].apply(lambda v: name_similarity(s_staff_raw, v))
        cand["_client_score"] = cand["Client Name"].apply(lambda v: name_similarity(s_client_raw, v))

        best = cand.sort_values(by=["_used","_staff_score","_client_score"], ascending=[True, False, False]).head(1)
        if best.empty:
            work.at[idx, "Reason"] = (row["Reason"] + ", No billing line match (date+staff+client)").strip(", ")
            work.at[idx, "Duration OK?"] = False
            continue

        staff_score = float(best["_staff_score"].iloc[0])
        client_score = float(best["_client_score"].iloc[0])
        if staff_score < STAFF_T or client_score < CLIENT_T:
            work.at[idx, "Reason"] = (row["Reason"] + ", No billing line match (date+staff+client)").strip(", ")
            work.at[idx, "Duration OK?"] = False
            continue

        if bool(best["_used"].iloc[0]):
            work.at[idx, "Reason"] = (row["Reason"] + ", Billing line already used").strip(", ")
            work.at[idx, "Duration OK?"] = False
            continue

        # mark used & record match
        b_idx = best.index[0]; bill.loc[b_idx, "_used"] = True
        b_minutes = float(best["_bill_minutes"].iloc[0]) if pd.notna(best["_bill_minutes"].iloc[0]) else None

        work.at[idx, "Billing matched staff"] = str(best["Staff Name"].iloc[0])
        work.at[idx, "Billing matched client"] = str(best["Client Name"].iloc[0])
        work.at[idx, "Billing date"] = best["_bill_date"].iloc[0]
        work.at[idx, "Billing hours (min)"] = b_minutes
        work.at[idx, "Matched staff score"] = round(staff_score, 3)
        work.at[idx, "Matched client score"] = round(client_score, 3)

        s_minutes = float(work.at[idx, "Session duration (min)"]) if pd.notna(work.at[idx, "Session duration (min)"]) else None
        if s_minutes is None or b_minutes is None:
            work.at[idx, "Reason"] = (row["Reason"] + ", Missing duration/billing minutes").strip(", ")
            work.at[idx, "Duration OK?"] = False
            continue

        diff = abs(s_minutes - b_minutes)
        work.at[idx, "Duration diff (min)"] = round(diff, 2)
        ok = diff <= TOL_MIN
        work.at[idx, "Duration OK?"] = bool(ok)
        if not ok:
            work.at[idx, "Reason"] = (row["Reason"] + f", Duration mismatch (>±{TOL_MIN} min)").strip(", ")

# =============== Geolocation (always runs, independent of duration) ===============
st.divider()
st.subheader("Geolocation pass")
if not MAPBOX_TOKEN:
    st.warning("No Mapbox token detected in environment or secrets (MAPBOX_ACCESS_TOKEN). Geocoding will be skipped.")

need_client_keys = set()
for _, row in work.iterrows():
    u_ll = user_coords[_]; p_ll = parent_coords[_]
    if (u_ll is not None) or (p_ll is not None):
        primary_addr, _unit = split_primary_unit(row.get(COL_CLIENT_ADDR))
        key = (
            primary_addr.strip(),
            str(row.get(COL_CLIENT_CITY) or "").strip(),
            DEFAULT_STATE_CLEAN,
            clean_zip(row.get(COL_CLIENT_ZIP)),
        )
        need_client_keys.add(key)

fwd_cache = {}
if MAPBOX_TOKEN and need_client_keys:
    prog = st.progress(0.0); total = len(need_client_keys)
    for i, key in enumerate(need_client_keys, 1):
        lat, lon, note, q_used, meta = robust_forward(*key, token=MAPBOX_TOKEN)
        fwd_cache[key] = (
            (lat, lon) if lat is not None and lon is not None else None,
            note, q_used, meta,
        )
        prog.progress(i/total); time.sleep(0.02)

flagged_mask = []
for idx, row in work.iterrows():
    r_list = [x for x in str(work.at[idx, "Reason"]).split(", ") if x] if work.at[idx, "Reason"] else []
    u_ll = user_coords[idx]; p_ll = parent_coords[idx]

    if (u_ll is None) and (p_ll is None):
        work.at[idx, "Geo OK?"] = False if any("present but no location" in x for x in r_list) else None
        flagged_mask.append(bool(r_list))
        continue

    primary_addr, _unit = split_primary_unit(row.get(COL_CLIENT_ADDR))
    key = (
        primary_addr.strip(),
        str(row.get(COL_CLIENT_CITY) or "").strip(),
        DEFAULT_STATE_CLEAN,
        clean_zip(row.get(COL_CLIENT_ZIP)),
    )
    client_ll, _note, _q, _meta = fwd_cache.get(key, (None, "NO_GEOCODE", "", {}))
    if client_ll is None:
        r_list.append("Unable to geocode client address")
        work.at[idx, "Reason"] = ", ".join(sorted(set(r_list)))
        work.at[idx, "Geo OK?"] = False
        flagged_mask.append(True)
        continue

    work.at[idx, "Client lat,lon"] = f"{client_ll[0]:.6f}, {client_ll[1]:.6f}"

    too_far = False
    if u_ll is not None:
        d_user = haversine_m(client_ll, u_ll)
        if d_user is not None and d_user > DISTANCE_METERS_THRESHOLD:
            r_list.append(f"User signature > {DISTANCE_FEET_THRESHOLD} ft from client"); too_far = True
    if p_ll is not None:
        d_parent = haversine_m(client_ll, p_ll)
        if d_parent is not None and d_parent > DISTANCE_METERS_THRESHOLD:
            r_list.append(f"Parent signature > {DISTANCE_FEET_THRESHOLD} ft from client"); too_far = True

    work.at[idx, "Reason"] = ", ".join(sorted(set(r_list)))
    work.at[idx, "Geo OK?"] = bool(not too_far)
    flagged_mask.append(bool(r_list))

# Reverse geocode only flagged to save calls
rows_to_reverse = work.index[flagged_mask]
rev_cache = {}
if MAPBOX_TOKEN and len(rows_to_reverse) > 0:
    need_rev = set()
    for idx in rows_to_reverse:
        u_ll = user_coords[idx]; p_ll = parent_coords[idx]
        if u_ll: need_rev.add(("user", round(u_ll[0], 6), round(u_ll[1], 6)))
        if p_ll: need_rev.add(("parent", round(p_ll[0], 6), round(p_ll[1], 6)))
    prog = st.progress(0.0); total = len(need_rev) if need_rev else 1
    for i, key in enumerate(need_rev or [], 1):
        _, lat, lon = key
        rev_cache[key] = _mapbox_reverse(lat, lon, MAPBOX_TOKEN)
        prog.progress(i/max(total,1)); time.sleep(0.02)

for idx in rows_to_reverse:
    u_ll = user_coords[idx]; p_ll = parent_coords[idx]
    if u_ll:
        key = ("user", round(u_ll[0], 6), round(u_ll[1], 6))
        work.at[idx, "User signature address"] = rev_cache.get(key, work.at[idx, "User signature address"])
    if p_ll:
        key = ("parent", round(p_ll[0], 6), round(p_ll[1], 6))
        work.at[idx, "Parent signature address"] = rev_cache.get(key, work.at[idx, "Parent signature address"])

# =============== Issue typing & Tabs ===============
def issue_type(row):
    d = row.get("Duration OK?")
    g = row.get("Geo OK?")
    if d is False and g is False: return "Both"
    if d is False: return "Duration"
    if g is False: return "Geolocation"
    return "None"

work["Issue Type"] = work.apply(issue_type, axis=1)
flagged = work[work["Reason"].astype(str).str.strip() != ""].copy()
clean = work[work["Reason"].astype(str).str.strip() == ""].copy()

tabs = st.tabs(["Overview", "Flagged (needs attention)", "Clean (no issues)", "Debug"])
with tabs[0]:
    st.metric("Total filtered sessions", len(work))
    st.metric("Flagged", len(flagged))
    st.metric("Clean", len(clean))

with tabs[1]:
    out_cols = [c for c in END_OUTPUT_COLS if c in flagged.columns]
    st.dataframe(flagged[out_cols], use_container_width=True)
    st.download_button(
        "⬇️ Download FLAGGED rows (CSV)",
        data=flagged[out_cols].to_csv(index=False).encode("utf-8"),
        file_name="flagged_combined_checks.csv",
        mime="text/csv",
    )

with tabs[2]:
    out_cols_c = [c for c in END_OUTPUT_COLS if c in clean.columns]
    st.dataframe(clean[out_cols_c], use_container_width=True)
    st.download_button(
        "⬇️ Download CLEAN rows (CSV)",
        data=clean[out_cols_c].to_csv(index=False).encode("utf-8"),
        file_name="clean_combined_checks.csv",
        mime="text/csv",
    )

with tabs[3]:
    # compact debug: show a small sample, plus unmatched billing lines (if billing uploaded)
    st.write("**Sample of working dataframe (first 50 rows)**")
    st.dataframe(work.head(50), use_container_width=True)
    if bill_file:
        unused = bill[~bill["_used"]].copy()
        st.write("**Unused Billing Lines**")
        st.dataframe(unused[["Staff Name", "Client Name", "Appt. Date", "Billing Hours"]], use_container_width=True)
