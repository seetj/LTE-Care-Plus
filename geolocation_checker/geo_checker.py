# aba_matcher_time_first.py
# Streamlit app to match Sessions ↔ Billing using time-window first (no duration gate).
# Uses per-(Date, Client) Hungarian assignment. Duration only for post-match flags.
import math
import re
from difflib import SequenceMatcher
from datetime import datetime, date, time as dtime, timedelta
import os
import requests
import pandas as pd
import streamlit as st

# =========================
# App / Tuning Parameters
# =========================
st.set_page_config(page_title="AlohaABA — Time-First Matcher (Per Date + Client)", layout="wide")
st.title("AlohaABA — Time-First Session↔Billing Matcher (Per Date + Client)")

# Billing columns (yours)
BILL_REQUIRED = [
    "Service Name", "Appt. Date", "Appt. Start Time", "Appt. End Time",
    "Staff Name", "Client Name", "Rendering Provider", "Completed",
    "Billing Hours", "Client City", "Patient Address"
]

# Session columns (yours)
SESS_REQUIRED = [
    "Status", "Client", "Date/Time", "Start time", "End time", "Duration", "User",
    "Activity type", "Session", "User signature", "User signature location",
    "Parent signature", "Parent signature location",
    "Client: Address Line 1", "Client: Zip / Postal Code", "Client: City / District"
]

# Filters
SESSION_REQUIRED_VALUE = "1:1 BT Direct Service"
STATUS_REQUIRED_VALUE  = "Transferred to AlohaABA"

# Thresholds
CLIENT_T = 0.85
STAFF_T  = 0.85
TIME_OVERLAP_MIN = 10.0        # require at least 10 min overlap OR allow by start-gap
TIME_START_MAX_GAP_MIN = 90.0  # if little/no overlap, allow if starts within 90 min
TOL_UNDER_MIN = 7.0            # flag if billing − session < −7 (under-billed)
OVER_BILL_TOL = 8.0            # flag if billing − session > 8 (over-billed)
BIG = 10**6

# Cost weights (NO duration term; duration only for flags)
W_START_GAP_HR = 0.8
W_END_GAP_HR   = 0.8
W_CSIM         = 0.15
W_SSIM         = 0.05
W_COMPLETED    = 2.0
W_OVERLAP      = 0.5       # reward for higher overlap ratio
W_CITY_MATCH   = 0.25      # soft bonus

# --- Geolocation settings ---
DISTANCE_FEET_THRESHOLD = 800
DISTANCE_METERS_THRESHOLD = DISTANCE_FEET_THRESHOLD * 0.3048

# Mapbox token (already stored)
MAPBOX_TOKEN = (
    st.secrets.get("MAPBOX_ACCESS_TOKEN")
    or os.getenv("MAPBOX_ACCESS_TOKEN")
    or None
)

# =========================
# Utilities
# =========================
def read_any(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        try:
            return pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0); return pd.read_csv(uploaded, encoding="latin-1")
    return pd.read_excel(uploaded)

try:
    from unidecode import unidecode
except Exception:
    unidecode = lambda s: s

SUFFIXES = {"jr","sr","ii","iii","iv","v"}

def norm_name(s: str) -> str:
    if s is None: return ""
    s = unidecode(str(s).strip())
    if "," in s:  # "Last, First, Suffix"
        parts = [p.strip() for p in s.split(",") if p is not None]
        last = parts[0] if len(parts)>=1 else ""
        first_middle = parts[1] if len(parts)>=2 else ""
        suffix = parts[2] if len(parts)>=3 else ""
        s = f"{first_middle} {last} {suffix}".strip()
    s = re.sub(r"[^A-Za-z\s]", " ", s)
    toks = re.sub(r"\s+", " ", s).strip().lower().split()
    toks = [t for t in toks if len(t)>1 and t not in SUFFIXES]
    return " ".join(toks)

def _basic_tokens(raw: str) -> str:
    s = unidecode(str(raw or "").strip().lower())
    s = re.sub(r"[^a-z\s,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

SURNAME_PREFIXES = {"de","del","dela","de la","da","dos","das","di","van","von","bin","ibn","al","el","mc","mac","la","le"}

def _split_given_surname(raw: str):
    s = _basic_tokens(raw)
    if not s: return [], []
    if "," in s:
        last, rest = [p.strip() for p in s.split(",", 1)]
        given = rest.split() if rest else []
        surname = last.split() if last else []
        return given, surname
    toks = s.split()
    if not toks: return [], []
    if len(toks) >= 2 and toks[-2] in SURNAME_PREFIXES:
        surname = toks[-2:]; given = toks[:-2]
    else:
        surname = toks[-1:]; given = toks[:-1]
    return given, surname

def _join_tokens(tokens): return " ".join(tokens), "".join(tokens)

def _token_set_score(a_tokens, b_tokens) -> float:
    ta, tb = set(a_tokens), set(b_tokens)
    if not ta or not tb: return 0.0
    overlap = len(ta & tb)
    return (2.0 * overlap) / (len(ta) + len(tb))

def name_similarity(a_raw: str, b_raw: str) -> float:
    ga, sa = _split_given_surname(a_raw); gb, sb = _split_given_surname(b_raw)
    base = _token_set_score(ga + sa, gb + sb)
    if not (ga or sa) or not (gb or sb): return base
    _, givenA_j = _join_tokens(ga); _, givenB_j = _join_tokens(gb)
    surA = set(_join_tokens(sa)); surB = set(_join_tokens(sb))
    fullA = givenA_j + "".join(sa); fullB = givenB_j + "".join(sb)
    if fullA and fullB and fullA == fullB: return 1.0
    if bool(set(surA) & set(surB)):
        from difflib import SequenceMatcher
        if givenA_j and givenB_j:
            if givenA_j == givenB_j: return 1.0
            if SequenceMatcher(None, givenA_j, givenB_j).ratio() >= 0.90:
                base = max(base, 0.94)
        base = min(1.0, base + 0.15)
    a_single = len(_basic_tokens(a_raw).split()) == 1
    b_single = len(_basic_tokens(b_raw).split()) == 1
    if a_single and not b_single and _basic_tokens(a_raw) == fullB: return 1.0
    if b_single and not a_single and _basic_tokens(b_raw) == fullA: return 1.0
    return base

def parse_time_maybe(s):
    if s is None or (isinstance(s, float) and math.isnan(s)): return None
    txt = str(s).strip()
    if not txt: return None
    for fmt in ["%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M%p", "%H%M"]:
        try:
            return datetime.strptime(txt, fmt).time()
        except Exception:
            pass
    try:
        v = float(txt)
        if 0 <= v < 1:
            seconds = int(round(v * 24 * 3600))
            return (datetime.min + timedelta(seconds=seconds)).time()
    except Exception:
        pass
    return None

def parse_date_maybe(s):
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None

def parse_hhmm_to_minutes(v) -> float:
    if v is None: return 0.0
    s = str(v).strip()
    if not s: return 0.0
    if ":" not in s:
        try: return float(s)
        except Exception: return 0.0
    parts = s.split(":")
    try:
        h = int(parts[0]); m = int(parts[1]) if len(parts)>1 else 0
        s_ = int(parts[2]) if len(parts)>2 else 0
        return float(h*60 + m) + (s_/60.0)
    except Exception:
        return 0.0

def detect_bill_units(series) -> str:
    as_str = series.astype(str)
    if as_str.str.contains(":").any(): return "HHMM"
    vals = pd.to_numeric(as_str.str.strip().str.replace(",", ""), errors="coerce")
    if (vals > 12).any(): return "MIN"
    return "HRS"

def to_minutes_session(val) -> float:  # used only for flags
    return parse_hhmm_to_minutes(val)

def to_minutes_billing(val, detected: str) -> float:
    if detected == "HHMM": return parse_hhmm_to_minutes(val)
    if detected == "MIN":
        try: return float(str(val).strip())
        except Exception: return 0.0
    try: return float(str(val).strip()) * 60.0
    except Exception: return 0.0

def minutes_between_times(t1: dtime, t2: dtime) -> float:
    if not t1 or not t2: return None
    dt1 = datetime.combine(date(2000,1,1), t1)
    dt2 = datetime.combine(date(2000,1,1), t2)
    return abs((dt2 - dt1).total_seconds()) / 60.0

def interval_overlap(a_start, a_end, b_start, b_end):
    if not a_start or not b_start:
        return (0.0, None, 0.0)
    if not a_end:
        a_end = (datetime.combine(date(2000,1,1), a_start) + timedelta(minutes=1)).time()
    if not b_end:
        b_end = (datetime.combine(date(2000,1,1), b_start) + timedelta(minutes=1)).time()
    dt = lambda t: datetime.combine(date(2000,1,1), t)
    a1, a2, b1, b2 = dt(a_start), dt(a_end), dt(b_start), dt(b_end)
    if a2 <= a1 or b2 <= b1:
        return (0.0, None, 0.0)
    overlap = max(0.0, (min(a2,b2) - max(a1,b1)).total_seconds()/60.0)
    union = (max(a2,b2) - min(a1,b1)).total_seconds()/60.0
    ratio = (overlap/union) if union > 0 else 0.0
    return (overlap, union, ratio)

def is_completed(v) -> bool:
    s = str(v).strip().lower()
    return s in {"yes","y","true","1","completed","done"}

# ---- Smart staff equivalence (handles joined-pinyin + first-name-only) ----
_CN_SURNAME_1 = {
    "li","liu","lin","wang","zhang","chen","huang","wu","xu","gao","guo","zhou","yang","sun",
    "zhao","ma","he","lu","luo","deng","qian","xiao","shi","hu","pan","fan","tang","song","cai"
}
# two-char surnames (joined)
_CN_SURNAME_2 = {"ouyang","sima","shangguan","zhugong","zhug e","zhugong"}  # keep small; joined matching only

def _split_joined_pinyin(token: str):
    """
    If 'liuhuidan' style, try to split into ('liu','huidan').
    Returns (surname, given) or (None, None) if not detected.
    """
    t = _basic_tokens(token).replace(" ", "")
    if not t or not t.isalpha() or len(t) < 5:
        return (None, None)
    # 2-char surname first
    for s2 in sorted(_CN_SURNAME_2, key=len, reverse=True):
        if t.startswith(s2):
            return (s2, t[len(s2):] or None)
    # 1-char surname list (actually many 1-romanized-char but treat as word in pinyin)
    for s1 in sorted(_CN_SURNAME_1, key=len, reverse=True):
        if t.startswith(s1) and len(t) > len(s1)+1:
            return (s1, t[len(s1):])
    return (None, None)

def _variants_from_joined(token: str):
    """
    For 'liuhuidan' → {'liu huidan','huidan liu','liu, huidan','huidanliu'} (normalized).
    """
    s,g = _split_joined_pinyin(token)
    if not s or not g: return set()
    v = {
        norm_name(f"{s} {g}"),
        norm_name(f"{g} {s}"),
        norm_name(f"{s}, {g}"),
        norm_name(f"{g}{s}"),
    }
    return v

def _variants_from_structured(fullname: str):
    """
    For 'Liu, Huidan' or 'Huidan Liu' → include a joined 'liuhuidan' variant.
    """
    raw = _basic_tokens(fullname)
    if "," in raw:
        last, first = [p.strip() for p in raw.split(",", 1)]
        given = first.split()
        surname = last.split()
    else:
        toks = raw.split()
        surname = toks[-1:] if toks else []
        given   = toks[:-1]   if toks else []
    given_join = "".join(given)
    surname_join = "".join(surname)
    if not given_join or not surname_join:
        return set()
    joined = norm_name(surname_join + given_join)  # 'liuhuidan'
    return {joined}

def staff_equiv(session_staff: str, billing_staff: str) -> float:
    """
    Smart name equivalence for staff (handles:
     - reversed first/last (e.g., 'Farsam' ↔ 'Abbas, Farsam')
     - middle initials and extra tokens
     - hyphenated last names
     - joined pinyin (e.g., 'liuhuidan' ↔ 'Liu, Huidan')
     - single-token fallback (e.g., 'Destini', 'Farsam')
    """
    from difflib import SequenceMatcher
    import re
    from unidecode import unidecode

    def _basic_tokens(s):
        s = unidecode(str(s or "").strip().lower())
        s = re.sub(r"[^a-z\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _variants_from_joined(x: str):
        x = unidecode(str(x).lower().strip())
        if len(x) <= 4:
            return set()
        out = set()
        for k in range(2, len(x) - 1):
            out.add(f"{x[:k]} {x[k:]}")
        return out

    def _variants_from_structured(x: str):
        parts = re.split(r"[,\s]+", unidecode(str(x).lower().strip()))
        joined = "".join(parts)
        return {joined, " ".join(parts)}

    s_norm = norm_name(session_staff)
    b_norm = norm_name(billing_staff)
    if not s_norm or not b_norm:
        return 0.0

    if s_norm == b_norm:
        return 1.0

    s_tokens = _basic_tokens(session_staff).split()
    b_tokens = _basic_tokens(billing_staff).split()
    if not s_tokens or not b_tokens:
        return 0.0

    # Reordered or overlapping names
    if any(t in b_norm for t in s_tokens) and any(t in s_norm for t in b_tokens):
        sim_ratio = SequenceMatcher(None, s_norm, b_norm).ratio()
        if sim_ratio >= 0.75:
            return 1.0

    # Swapped first/last
    s_rev = " ".join(reversed(s_tokens))
    if norm_name(s_rev) == b_norm:
        return 1.0

    # Joined-pinyin
    s_join_variants = _variants_from_joined(session_staff)
    b_join_variants = _variants_from_joined(billing_staff)
    s_struct_as_join = _variants_from_structured(session_staff)
    b_struct_as_join = _variants_from_structured(billing_staff)
    if (b_norm in s_join_variants) or (s_norm in b_join_variants):
        return 1.0
    if (b_norm in s_struct_as_join) or (s_norm in b_struct_as_join):
        return 1.0

    # Hyphen / spacing differences
    b_clean = b_norm.replace("-", " ")
    s_clean = s_norm.replace("-", " ")
    if SequenceMatcher(None, s_clean, b_clean).ratio() >= 0.9:
        return 1.0

    # First-name-only fallback
    if len(s_tokens) == 1 and s_tokens[0] in b_tokens:
        return 0.9
    if len(b_tokens) == 1 and b_tokens[0] in s_tokens:
        return 0.9

    # Generic fuzzy fallback
    base = name_similarity(session_staff, billing_staff)
    if base >= 0.85:
        return 1.0 if base > 0.9 else base
    return base

# Simple Hungarian-like solver (OK for small matrices)
def hungarian_min_cost(cost_df: pd.DataFrame):
    A = cost_df.copy().values.astype(float)
    A[~(A < BIG)] = BIG
    rmin = A.min(axis=1, initial=0); rmin[rmin >= BIG] = 0; A = (A.T - rmin).T
    cmin = A.min(axis=0, initial=0); cmin[cmin >= BIG] = 0; A = A - cmin
    n_rows, n_cols = A.shape
    assigned_cols, assignment = set(), {}
    for _ in range(n_rows):
        zero_counts = [(i, (A[i] == 0).sum()) for i in range(n_rows) if i not in assignment]
        if not zero_counts: break
        zero_counts.sort(key=lambda x: x[1])
        i = zero_counts[0][0]
        zero_cols = [j for j in range(n_cols) if A[i, j] == 0 and j not in assigned_cols]
        if zero_cols:
            j = zero_cols[0]; assignment[i] = j; assigned_cols.add(j)
        else:
            j = int(A[i].argmin())
            if A[i, j] >= BIG: assignment[i] = None
            else: assignment[i] = j; assigned_cols.add(j)
    for i in range(n_rows):
        if i in assignment: continue
        j = int(A[i].argmin())
        if A[i, j] >= BIG: assignment[i] = None
        else:
            if j in assigned_cols: assignment[i] = None
            else: assignment[i] = j; assigned_cols.add(j)
    return assignment, assigned_cols

# ---- Geo helpers ----
def parse_latlon(s: str):
    if s is None or (isinstance(s, float) and math.isnan(s)): return None
    txt = str(s).strip().replace("(", "").replace(")", "")
    if "," not in txt: return None
    try:
        lat, lon = [float(p.strip()) for p in txt.split(",", 1)]
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
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

def _mapbox_forward_one(q: str, token: str, timeout=5):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{q}.json"
    params = {"access_token": token, "limit": 1, "country": "US", "types": "address", "autocomplete": "false"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        js = r.json(); feats = js.get("features", [])
        if feats:
            lon, lat = feats[0].get("center", [None, None])
            if lat is not None and lon is not None:
                return (lat, lon)
    except Exception:
        pass
    return None

def forward_candidates(addr: str, city: str, state="NY"):
    addr = str(addr or "").strip()
    city = str(city or "").strip()
    parts = [p for p in [addr, city, state] if p]
    q1 = ", ".join(parts)
    q2 = f"{addr} {city} {state}".strip()
    return list(dict.fromkeys([q1, q2, addr]))  # unique in order

def geocode_patient_address(addr: str, city: str, token: str):
    for q in forward_candidates(addr, city):
        qp = requests.utils.quote(q)
        ll = _mapbox_forward_one(qp, token)
        if ll: return ll
    return None

# =========================
# UI — Uploads
# =========================
c1, c2 = st.columns(2)
with c1:
    sess_file = st.file_uploader("Upload Sessions (CSV/XLSX)", type=["csv","xlsx","xls"])
with c2:
    bill_file = st.file_uploader("Upload Billing (CSV/XLSX)", type=["csv","xlsx","xls"])

if not sess_file or not bill_file:
    st.info("Upload both files to begin."); st.stop()

sessions = read_any(sess_file)
billing  = read_any(bill_file)

missing_s = [c for c in SESS_REQUIRED if c not in sessions.columns]
missing_b = [c for c in BILL_REQUIRED if c not in billing.columns]
if missing_s: st.error(f"Sessions: missing columns {missing_s}"); st.stop()
if missing_b: st.error(f"Billing: missing columns {missing_b}"); st.stop()

st.caption("Preview — Sessions (first 20)"); st.dataframe(sessions.head(20), use_container_width=True)
st.caption("Preview — Billing (first 20)");  st.dataframe(billing.head(20), use_container_width=True)

# =========================
# Preprocess — Sessions
# =========================
work = sessions.copy()

# Filter to required Status + Service/Session
svc_mask = (
    (work["Status"].astype(str).str.strip() == STATUS_REQUIRED_VALUE) &
    (
        (work.get("Activity type", "").astype(str).str.strip() == SESSION_REQUIRED_VALUE) |
        (work.get("Session", "").astype(str).str.strip() == SESSION_REQUIRED_VALUE)
    )
)
work = work[svc_mask].copy()
work = work[work["Client"].astype(str).str.strip().str.lower() != "marry wang demo"].copy()
if work.empty:
    st.warning("No sessions match the required Status and Service/Session filters."); st.stop()

# Normalize names
work["_client_norm"] = work["Client"].astype(str).map(norm_name)
work["_staff_norm"]  = work["User"].astype(str).map(norm_name)

# Date / times
work["_sess_date"]   = pd.to_datetime(work["Date/Time"], errors="coerce").dt.date
work["_sess_start"]  = work["Start time"].apply(parse_time_maybe)
work["_sess_end"]    = work["End time"].apply(parse_time_maybe)

# Duration (only for flags)
work["_sess_dur_min"] = work["Duration"].apply(to_minutes_session)

# =========================
# Preprocess — Billing
# =========================
bill = billing.copy()
bill["_bill_date"]   = bill["Appt. Date"].apply(parse_date_maybe)
bill["_client_norm"] = bill["Client Name"].astype(str).map(norm_name)
bill["_staff_norm"]  = bill["Staff Name"].astype(str).map(norm_name)
bill["_completed"]   = bill["Completed"].apply(is_completed)

detected = detect_bill_units(bill["Billing Hours"])
bill["_bill_min"]   = bill["Billing Hours"].apply(lambda v: to_minutes_billing(v, detected))
bill["_start_t"]    = bill["Appt. Start Time"].apply(parse_time_maybe)
bill["_end_t"]      = bill["Appt. End Time"].apply(parse_time_maybe)

# Precompute patient lat/lon for all billing rows (cache by (Patient Address, Client City))
patient_ll_cache = {}
if MAPBOX_TOKEN:
    uniq_keys = {(str(row["Patient Address"]).strip(), str(row["Client City"]).strip())
                 for _, row in bill.iterrows()
                 if pd.notna(row["Patient Address"]) or pd.notna(row["Client City"])}

    pb = st.progress(0.0)
    total = len(uniq_keys) if uniq_keys else 1
    for idx, (addr, city) in enumerate(uniq_keys):
        if (addr, city) not in patient_ll_cache:
            patient_ll_cache[(addr, city)] = geocode_patient_address(addr, city, MAPBOX_TOKEN)
        pb.progress((idx+1)/total)
else:
    st.warning("No Mapbox token found. Geolocation checks will be skipped.")

# =========================
# Assignment (per Date+Client)
# =========================
results = []
pair_logs = []
unmatched = []

dates = sorted({d for d in work["_sess_date"].dropna()})
prog = st.progress(0.0)

for di, d in enumerate(dates, 1):
    sess_d = work[work["_sess_date"] == d].copy()
    bill_d = bill[bill["_bill_date"] == d].copy()
    if sess_d.empty:
        prog.progress(di/len(dates)); continue

    # For each session client on this date, find candidate billing rows with sufficient client similarity
    for s_client in sess_d["Client"].unique().tolist():
        s_rows = sess_d[sess_d["Client"] == s_client].copy()
        b_rows = bill_d[bill_d["Client Name"].apply(lambda c: name_similarity(s_client, c) >= CLIENT_T)].copy()

        if s_rows.empty: continue
        if b_rows.empty:
            for i in s_rows.index:
                unmatched.append({"Date": d, "SessIdx": int(i), "Client": s_client, "Why": "No billing lines for client/date"})
            continue

        s_idx = list(s_rows.index)
        b_idx = list(b_rows.index)
        cost_rows, meta_rows = [], []

        for i in s_idx:
            row_costs, row_meta = [], []
            s_staff  = work.at[i, "User"]
            s_start  = work.at[i, "_sess_start"]
            s_end    = work.at[i, "_sess_end"]
            s_dur    = float(work.at[i, "_sess_dur_min"] or 0.0)

            for j in b_idx:
                b_staff = bill.at[j, "Staff Name"]
                b_start = bill.at[j, "_start_t"]
                b_end   = bill.at[j, "_end_t"]
                b_dur   = float(bill.at[j, "_bill_min"] or 0.0)
                completed = bool(bill.at[j, "_completed"])

                c_score = name_similarity(work.at[i,"Client"], bill.at[j,"Client Name"])
                s_score = staff_equiv(s_staff, b_staff)

                # ---- Gates (NO duration gate) ----
                feasible = True
                reasons = []

                if c_score < CLIENT_T:
                    feasible = False; reasons.append(f"client_sim {c_score:.2f}<{CLIENT_T}")

                if s_score < STAFF_T:
                    feasible = False; reasons.append(f"staff_sim {s_score:.2f}<{STAFF_T}")

                overlap_min, union_min, overlap_ratio = interval_overlap(s_start, s_end, b_start, b_end)
                start_gap_min = minutes_between_times(s_start, b_start)

                if (overlap_min < TIME_OVERLAP_MIN) and (start_gap_min is not None and start_gap_min > TIME_START_MAX_GAP_MIN):
                    feasible = False; reasons.append("time windows too far (overlap<10m and start_gap>90m)")

                # Pair diagnostics (even if infeasible)
                pair_logs.append({
                    "Date": d, "SessIdx": int(i), "BillIdx": int(j),
                    "S_Client": work.at[i,"Client"], "B_Client": bill.at[j,"Client Name"],
                    "S_Staff": s_staff, "B_Staff": b_staff,
                    "S_Start": s_start, "S_End": s_end, "B_Start": b_start, "B_End": b_end,
                    "S_Dur": s_dur, "B_Dur": b_dur,
                    "ClientSim": round(c_score,3), "StaffSim": round(s_score,3),
                    "Completed": completed, "Feasible": feasible,
                    "Reason": "; ".join(reasons) if reasons else "",
                    "OverlapRatio": round(overlap_ratio,3), "StartGapMin": start_gap_min
                })

                if not feasible:
                    row_costs.append(BIG)
                    row_meta.append((c_score, s_score, b_dur, completed, reasons, overlap_ratio, start_gap_min))
                    continue

                # ---- Cost (no duration term) ----
                start_gap_hr = (start_gap_min/60.0) if (start_gap_min is not None) else 0.0
                end_gap_min  = minutes_between_times(s_end, b_end)
                end_gap_hr   = (end_gap_min/60.0) if (end_gap_min is not None) else 0.0

                city_match = False
                if pd.notna(work.at[i,"Client: City / District"]) and pd.notna(bill.at[j,"Client City"]):
                    city_match = str(work.at[i,"Client: City / District"]).strip().lower() == str(bill.at[j,"Client City"]).strip().lower()

                cost = (
                    W_START_GAP_HR * start_gap_hr +
                    (W_END_GAP_HR * end_gap_hr if end_gap_hr is not None else 0.0) -
                    W_CSIM * c_score -
                    W_SSIM * s_score -
                    (W_COMPLETED if completed else 0.0) -
                    (W_OVERLAP * overlap_ratio) -
                    (W_CITY_MATCH if city_match else 0.0)
                )
                if cost < 0: cost = 0.0

                row_costs.append(cost)
                row_meta.append((c_score, s_score, b_dur, completed, [], overlap_ratio, start_gap_min))

            cost_rows.append(row_costs)
            meta_rows.append(row_meta)

        # Pad to square and solve
        cost_df = pd.DataFrame(cost_rows, index=s_idx, columns=b_idx)
        S, B = cost_df.shape
        if S != B:
            if S < B:
                for k in range(B - S): cost_df.loc[f"_pad_{k}"] = [BIG]*B
            else:
                for k in range(S - B): cost_df[f"_pad_{k}"] = [BIG]*S

        assign, _ = hungarian_min_cost(cost_df)

        # Apply
        for r_pos, c_pos in assign.items():
            i_lab = cost_df.index[r_pos]
            if isinstance(i_lab, str) and i_lab.startswith("_pad_"): continue
            if c_pos is None:
                unmatched.append({"Date": d, "SessIdx": int(i_lab), "Client": s_client, "Why": "No feasible match"})
                continue
            j_lab = cost_df.columns[c_pos]
            if isinstance(j_lab, str) and str(j_lab).startswith("_pad_"):
                unmatched.append({"Date": d, "SessIdx": int(i_lab), "Client": s_client, "Why": "No feasible match"})
                continue
            if float(cost_df.iat[r_pos, c_pos]) >= BIG:
                unmatched.append({"Date": d, "SessIdx": int(i_lab), "Client": s_client, "Why": "No feasible match"})
                continue

            # meta
            s_row = s_idx.index(i_lab); b_col = b_idx.index(j_lab)
            c_score, s_score, b_dur, completed, _reasons, overlap_ratio, start_gap_min = meta_rows[s_row][b_col]
            s_dur = float(work.at[i_lab, "_sess_dur_min"] or 0.0)

            # --- Billing vs Session comparison: diff > 0 => Over-billed; diff < 0 => Under-billed ---
            s_minutes = float(s_dur or 0.0)
            b_minutes = float(b_dur or 0.0)
            diff = b_minutes - s_minutes
            over_flag  = diff > OVER_BILL_TOL
            under_flag = diff < -TOL_UNDER_MIN
            duration_ok = not (over_flag or under_flag)

            # --- Geolocation gate ---
            geo_ok = True
            geo_msgs = []

            u_sig_txt = str(work.at[i_lab, "User signature"]).strip()
            p_sig_txt = str(work.at[i_lab, "Parent signature"]).strip()
            if not u_sig_txt:
                geo_ok = False; geo_msgs.append("No user signature")
            if not p_sig_txt:
                geo_ok = False; geo_msgs.append("No parent signature")

            u_ll = parse_latlon(work.at[i_lab, "User signature location"]) if u_sig_txt else None
            p_ll = parse_latlon(work.at[i_lab, "Parent signature location"]) if p_sig_txt else None
            if u_sig_txt and not u_ll:
                geo_ok = False; geo_msgs.append("User signature present but no location")
            if p_sig_txt and not p_ll:
                geo_ok = False; geo_msgs.append("Parent signature present but no location")

            pat_addr = str(bill.at[j_lab, "Patient Address"]).strip()
            pat_city = str(bill.at[j_lab, "Client City"]).strip()
            client_ll = patient_ll_cache.get((pat_addr, pat_city)) if MAPBOX_TOKEN else None

            d_user_ft = None; d_parent_ft = None
            if (u_ll or p_ll):
                if not client_ll and MAPBOX_TOKEN:
                    geo_ok = False
                    geo_msgs.append("Unable to geocode patient address")
                elif client_ll:
                    if u_ll:
                        d_user_m = haversine_m(client_ll, u_ll)
                        if d_user_m is not None:
                            d_user_ft = round(d_user_m / 0.3048, 1)
                            if d_user_ft > DISTANCE_FEET_THRESHOLD:
                                geo_ok = False
                                geo_msgs.append(f"User signature {int(round(d_user_ft))} ft from patient (> {DISTANCE_FEET_THRESHOLD} ft)")
                    if p_ll:
                        d_parent_m = haversine_m(client_ll, p_ll)
                        if d_parent_m is not None:
                            d_parent_ft = round(d_parent_m / 0.3048, 1)
                            if d_parent_ft > DISTANCE_FEET_THRESHOLD:
                                geo_ok = False
                                geo_msgs.append(f"Parent signature {int(round(d_parent_ft))} ft from patient (> {DISTANCE_FEET_THRESHOLD} ft)")

            # If no geo issues, provide a short OK note
            if not geo_msgs:
                geo_msgs.append("Geolocation OK")
            geo_details = "; ".join(geo_msgs)

            # Human-friendly duration verdict
            if under_flag and not over_flag:
                duration_verdict = f"HiRasmus session under-billed by {int(round(-diff))} min"
            elif over_flag and not under_flag:
                duration_verdict = f"HiRasmus session over-billed by {int(round(diff))} min"
            elif over_flag and under_flag:
                duration_verdict = "⚠️ Data conflict: check durations / duplicate match"
            else:
                duration_verdict = "OK"

            # Final reason text
            if duration_verdict == "OK" and geo_ok:
                final_reason = "OK"
            else:
                parts = []
                if duration_verdict != "OK":
                    parts.append(duration_verdict)
                if not geo_ok or geo_details != "Geolocation OK":
                    parts.append(geo_details)
                final_reason = "; ".join(parts) if parts else "OK"

            results.append({
                "Date": d,
                "SessionIdx": int(i_lab), "BillingIdx": int(j_lab),
                "Client": work.at[i_lab, "Client"],
                "BT": work.at[i_lab, "User"],
                "Billing Client": bill.at[j_lab, "Client Name"],
                "Billing Staff": bill.at[j_lab, "Staff Name"],
                "Service Name": bill.at[j_lab, "Service Name"],
                "Rendering Provider": bill.at[j_lab, "Rendering Provider"],
                "Session Start": work.at[i_lab, "_sess_start"],
                "Session End": work.at[i_lab, "_sess_end"],
                "Billing Start": bill.at[j_lab, "_start_t"],
                "Billing End": bill.at[j_lab, "_end_t"],
                "Session Duration (min)": s_minutes,
                "Billing Hours (min)": b_minutes,
                "Duration diff (min)": round(diff, 2),  # Billing − Session
                "Duration OK?": duration_ok,
                "Under-billed? (>tol)": under_flag,
                "Over-billed? (>tol)": over_flag,
                "Completed": "Yes" if completed else "No",
                "ClientSim": round(float(c_score),3),
                "StaffSim": round(float(s_score),3),
                "OverlapRatio": round(float(overlap_ratio or 0.0),3),
                "StartGapMin": start_gap_min,          # << unified name
                "Geo OK?": geo_ok,                    # << used for Issue Type
                "Geo Details": geo_details,
                "Reason": final_reason,               # single-line verdict for humans
            })

    prog.progress(di/len(dates))

# =========================
# Outputs
# =========================
res_df  = pd.DataFrame(results)
pair_df = pd.DataFrame(pair_logs)
unm_df  = pd.DataFrame(unmatched)

def issue_type(row):
    # Decide None / Duration / Geolocation / Both
    d = float(row.get("Duration diff (min)") or 0.0)  # Billing − Session
    geo_ok = bool(row.get("Geo OK?", True))
    over_bad  = d > OVER_BILL_TOL
    under_bad = d < -TOL_UNDER_MIN
    dur_bad = (over_bad or under_bad)
    if dur_bad and not geo_ok: return "Both"
    if dur_bad:                return "Duration"
    if not geo_ok:             return "Geolocation"
    return "None"

if not res_df.empty:
    res_df["Issue Type"] = res_df.apply(issue_type, axis=1)
    # Build clean & flagged DataFrames for export (robust mask)
    flagged_mask = res_df["Issue Type"].notna() & res_df["Issue Type"].ne("None")
    clean_df   = res_df[~flagged_mask].copy()
    flagged_df = res_df[flagged_mask].copy()
else:
    clean_df = pd.DataFrame()
    flagged_df = pd.DataFrame()

tabs = st.tabs(["Overview", "Matched (All)", "Flagged", "Unmatched", "Per-pair Diagnostics"])

with tabs[0]:
    st.metric("Sessions matched", len(res_df))
    st.metric("Flagged", int(flagged_df.shape[0]))
    st.metric("Unmatched sessions", len(unm_df))
    st.subheader("Quick Downloads")

# Downloads (shown under Overview heading)
if res_df.empty:
    st.info("No results to download yet.")
else:
    base_cols = [
        "Date","Client","BT","Billing Client","Billing Staff",
        "Service Name","Rendering Provider",
        "Session Start","Session End","Billing Start","Billing End",
        "Session Duration (min)","Billing Hours (min)","Duration diff (min)",
        "Issue Type","Reason","Geo Details","Completed",
        "ClientSim","StaffSim","OverlapRatio","StartGapMin"
    ]
    # Intersect with actual columns to avoid KeyErrors
    clean_cols   = [c for c in base_cols if c in clean_df.columns]
    flagged_cols = [c for c in base_cols if c in flagged_df.columns]

    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"✅ Clean cases: {len(clean_df)}")
        st.download_button(
            "⬇️ Download Clean CSV",
            data=(clean_df[clean_cols].to_csv(index=False).encode("utf-8") if not clean_df.empty else b""),
            file_name="clean_cases.csv",
            mime="text/csv",
            key="dl_clean_cases",
            disabled=clean_df.empty
        )
    with c2:
        st.caption(f"⚠️ Flagged cases: {len(flagged_df)}")
        st.download_button(
            "⬇️ Download Flagged CSV",
            data=(flagged_df[flagged_cols].to_csv(index=False).encode("utf-8") if not flagged_df.empty else b""),
            file_name="flagged_cases.csv",
            mime="text/csv",
            key="dl_flagged_cases",
            disabled=flagged_df.empty
        )

with tabs[1]:
    if res_df.empty:
        st.info("No matches.")
    else:
        show_cols = [
            "Date","Client","BT",
            "Service Name","Rendering Provider",
            "Billing Start","Billing End",
            "Session Duration (min)","Billing Hours (min)","Duration diff (min)",
            "Completed",
            "Issue Type","Reason","Geo Details"
        ]
        show_cols = [c for c in show_cols if c in res_df.columns]
        st.dataframe(res_df[show_cols], use_container_width=True)
        st.download_button(
            "Download Matched CSV",
            res_df[show_cols].to_csv(index=False).encode("utf-8"),
            "matched_time_first.csv",
            "text/csv"
        )

with tabs[2]:
    if flagged_df.empty:
        st.success("No flagged rows 🎉")
    else:
        show_cols = [
            "Date","Client","BT",
            "Billing Start","Billing End",
            "Session Duration (min)","Billing Hours (min)","Duration diff (min)",
            "Issue Type","Reason","Geo Details","Completed"
        ]
        show_cols = [c for c in show_cols if c in flagged_df.columns]
        st.dataframe(flagged_df[show_cols], use_container_width=True)
        st.download_button(
            "Download Flagged CSV",
            flagged_df[show_cols].to_csv(index=False).encode("utf-8"),
            "flagged_time_first.csv",
            "text/csv"
        )

with tabs[3]:
    if unm_df.empty:
        st.success("No unmatched sessions 🎉")
    else:
        st.dataframe(unm_df, use_container_width=True)
        st.download_button(
            "Download Unmatched CSV",
            unm_df.to_csv(index=False).encode("utf-8"),
            "unmatched_time_first.csv",
            "text/csv"
        )

with tabs[4]:
    if pair_df.empty:
        st.info("No diagnostics yet.")
    else:
        show_cols = ["Date","SessIdx","BillIdx","S_Client","B_Client",
                     "S_Staff","B_Staff","S_Start","S_End","B_Start","B_End",
                     "S_Dur","B_Dur","ClientSim","StaffSim","Completed",
                     "Feasible","Cost","Reason","OverlapRatio","StartGapMin"]
        show_cols = [c for c in show_cols if c in pair_df.columns]
        st.dataframe(pair_df[show_cols], use_container_width=True, height=420)
        st.download_button(
            "Download Pair Diagnostics",
            pair_df[show_cols].to_csv(index=False).encode("utf-8"),
            "pair_diagnostics_time_first.csv",
            "text/csv"
        )

