# app.py — Faster Mapbox version with simple messages & cleaner columns
# - Simple reasons: "No user signature" / "No parent signature" (no format wording)
# - Removed output columns: "User signature", "Parent signature"
# - Removed "Geocode Debug" column

import os
import re
import math
import time
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Signature Proximity Checker — Fast (Mapbox, 800 ft)", layout="wide")
st.title("Signature Proximity Checker — Fast (Mapbox, 800 ft)")

# ---------------- Sidebar: API key & settings ----------------
st.sidebar.header("🔑 API Keys")
mapbox_token_ui = st.sidebar.text_input("Mapbox Access Token", type="password", placeholder="pk.XXXX…")
MAPBOX_TOKEN = (
    os.getenv("MAPBOX_ACCESS_TOKEN")
    or st.secrets.get("MAPBOX_ACCESS_TOKEN", None)
    or (mapbox_token_ui.strip() if mapbox_token_ui else None)
)

if not MAPBOX_TOKEN:
    st.warning("No Mapbox token detected. Enter it in the sidebar or set MAPBOX_ACCESS_TOKEN.")

# ---------------- UI Controls ----------------
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
session_filter_value = st.text_input("Filter Session equals", value="1:1 BT Direct Service")

DISTANCE_FEET_THRESHOLD = 800
DISTANCE_METERS_THRESHOLD = DISTANCE_FEET_THRESHOLD * 0.3048

DEFAULT_STATE = st.text_input("Default State (2-letter)", value="NY")
reverse_only_flagged = st.checkbox("Reverse-geocode signature addresses only for flagged rows (faster)", value=True)

# Optional: forward-geocode debug toggle (kept; doesn't affect "geocode debug line" column)
verbose_forward_debug = st.sidebar.checkbox("Show forward-geocode queries (debug)", value=False)

COL_CLIENT_ADDR = "Client: Address Line 1"
COL_CLIENT_CITY = "Client: City / District"
COL_CLIENT_ZIP  = "Client: Zip / Postal Code"

REQUIRED_COLS = [
    "Client", "User", "Date/Time", "End time", "Duration", "Session",
    "User signature", "User signature location",
    "Parent signature", "Parent signature location",
    COL_CLIENT_ADDR, COL_CLIENT_CITY, COL_CLIENT_ZIP
]

# NOTE: removed "User signature" and "Parent signature" from visible columns,
# and removed "Geocode Debug".
VISIBLE_COLS = [
    "Client", "BT", "Date/Time", "End time", "Duration", "Session",
    COL_CLIENT_ADDR, COL_CLIENT_CITY, COL_CLIENT_ZIP,
    "Client lat,lon",
    "User signature location", "User signature address",
    "Parent signature location", "Parent signature address",
    "Forward geocode query",        # optional debug
    "Forward matching_text",        # optional debug
    "Reason"
]

# ---------------- Helpers ----------------
def read_any(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(f)
        except UnicodeDecodeError:
            f.seek(0); return pd.read_csv(f, encoding="latin-1")
    return pd.read_excel(f)

def validate_columns(df: pd.DataFrame):
    return [c for c in REQUIRED_COLS if c not in df.columns]

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
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(x), math.sqrt(1-x))

def clean_addr(addr: str) -> str:
    """Minimal cleaning: trim + collapse internal whitespace. Keep everything else."""
    if not addr:
        return ""
    s = str(addr)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _mapbox_forward(q: str, token: str, timeout=5):
    """Low-level forward geocode; returns (lat, lon) or (None, None)."""
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(q)}.json"
    params = {"access_token": token, "limit": 1, "country": "US"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        js = r.json()
        feats = js.get("features", [])
        if feats:
            lon, lat = feats[0]["center"]
            return (lat, lon)
    except Exception:
        return (None, None)
    return (None, None)

def _mapbox_forward_detail(q: str, token: str, timeout=5):
    """
    Returns (lat, lon, meta) where meta includes text, place_name, matching_text, relevance.
    If not found, returns (None, None, {}).
    """
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(q)}.json"
    params = {"access_token": token, "limit": 1, "country": "US"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        js = r.json()
        feats = js.get("features", [])
        if feats:
            f0 = feats[0]
            lon, lat = f0.get("center", [None, None])
            meta = {
                "text": f0.get("text", ""),
                "place_name": f0.get("place_name", ""),
                "matching_text": f0.get("matching_text", ""),
                "relevance": f0.get("relevance", None),
            }
            if lat is not None and lon is not None:
                return (lat, lon, meta)
    except Exception:
        pass
    return (None, None, {})

def _mapbox_reverse(lat, lon, token: str, timeout=5):
    """Low-level reverse geocode; returns place_name or ''."""
    if lat is None or lon is None: return ""
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{lon},{lat}.json"
    params = {"access_token": token, "limit": 1, "country": "US"}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        js = r.json()
        feats = js.get("features", [])
        if feats:
            return feats[0].get("place_name", "") or ""
    except Exception:
        return ""
    return ""

def forward_candidates(addr, city, state, zip_):
    # Minimal/strict: lead with the exact joined address; keep a few light fallbacks.
    a = clean_addr(addr); c = (city or "").strip(); s = (state or "").strip(); z = (zip_ or "").strip()
    return [
        ", ".join([x for x in [a, c, s, z] if x]),
        ", ".join([x for x in [a, c, s] if x]),
        a,
    ]

def robust_forward(addr, city, state, zip_, token, retries=1):
    """
    Tries candidates. Returns (lat, lon, note, q_used, meta)
    """
    tried = set()
    for q in forward_candidates(addr, city, state, zip_):
        q = q.strip().strip(",")
        if not q or q in tried:
            continue
        tried.add(q)

        lat, lon, meta = _mapbox_forward_detail(q, token)
        if lat is not None and lon is not None:
            note = f"OK: {q}"
            return (lat, lon, note, q, meta)

        if retries:
            lat, lon, meta = _mapbox_forward_detail(q, token)
            if lat is not None and lon is not None:
                note = f"OK(retry): {q}"
                return (lat, lon, note, q, meta)

    return (None, None, f"NO_MATCH after tries: {', '.join(list(tried)[:3])}…", "", {})

# ---------------- Main ----------------
if uploaded:
    df = read_any(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    missing = validate_columns(df)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    work = df[df["Session"].astype(str).str.strip() == session_filter_value].copy()
    work.rename(columns={"User": "BT"}, inplace=True)
    work.reset_index(drop=True, inplace=True)

    st.info("Scanning rows and preparing unique geocode tasks…")

    # First pass: compute reasons + collect unique client geocodes needed
    reasons = []
    need_client_keys = set()
    user_coords = []
    parent_coords = []

    for _, row in work.iterrows():
        r = []
        u_raw = row.get("User signature location")
        p_raw = row.get("Parent signature location")
        u_missing = (u_raw is None) or (str(u_raw).strip() == "")
        p_missing = (p_raw is None) or (str(p_raw).strip() == "")

        # Treat missing or malformed locations the same: simple messages only
        u_ll = parse_latlon(u_raw) if not u_missing else None
        p_ll = parse_latlon(p_raw) if not p_missing else None

        if u_ll is None:
            r.append("No user signature")
        if p_ll is None:
            r.append("No parent signature")

        user_coords.append(u_ll)
        parent_coords.append(p_ll)

        # Only geocode client if at least one coord parsed OK
        if (u_ll is not None) or (p_ll is not None):
            key = (
                str(row.get(COL_CLIENT_ADDR) or "").strip(),
                str(row.get(COL_CLIENT_CITY) or "").strip(),
                str(DEFAULT_STATE or "").strip(),
                str(row.get(COL_CLIENT_ZIP) or "").strip(),
            )
            need_client_keys.add(key)

        reasons.append(", ".join(r))

    work["Reason"] = reasons
    # Pre-fill columns so we can assign later
    work["Client lat,lon"] = ""
    work["User signature address"] = ""
    work["Parent signature address"] = ""
    work["Forward geocode query"] = ""     # optional debug
    work["Forward matching_text"] = ""     # optional debug

    # ------- Forward geocode (deduped) only for needed unique client keys -------
    st.info(f"Geocoding unique client addresses: {len(need_client_keys)} (deduped)")
    fwd_cache = {}
    if MAPBOX_TOKEN and need_client_keys:
        prog = st.progress(0.0)
        total = len(need_client_keys)
        for i, key in enumerate(need_client_keys, 1):
            lat, lon, note, q_used, meta = robust_forward(*key, token=MAPBOX_TOKEN)
            fwd_cache[key] = (
                (lat, lon) if lat is not None and lon is not None else None,
                note,
                q_used,
                meta,
            )
            if verbose_forward_debug:
                st.write(
                    f"[{i}/{total}] Query → **{q_used or '(none)'}** | "
                    f"Result: {('%.6f, %.6f' % (lat, lon)) if lat is not None else 'NO_MATCH'} | "
                    f"matching_text: **{meta.get('matching_text','')}** | "
                    f"text: {meta.get('text','')} | "
                    f"place_name: {meta.get('place_name','')} | "
                    f"relevance: {meta.get('relevance')}"
                )
            prog.progress(i/total)
            time.sleep(0.02)
    else:
        if not MAPBOX_TOKEN:
            st.warning("Skipping client geocoding — no Mapbox token provided.")

    # Optional: compact mapping of geocode attempts
    with st.expander("Forward-geocode debug table"):
        if fwd_cache:
            dbg_rows = []
            for (addr, city, state, zip_), (ll, note, q_used, meta) in fwd_cache.items():
                dbg_rows.append({
                    "Input Address": addr,
                    "Input City": city,
                    "Input State": state,
                    "Input Zip": zip_,
                    "Query Used": q_used,
                    "Result Lat,Lon": ("" if ll is None else f"{ll[0]:.6f}, {ll[1]:.6f}"),
                    "matching_text": meta.get("matching_text", ""),
                    "text": meta.get("text", ""),
                    "place_name": meta.get("place_name", ""),
                    "relevance": meta.get("relevance", ""),
                    "Note": note,
                })
            st.dataframe(pd.DataFrame(dbg_rows), use_container_width=True)
        else:
            st.write("No forward-geocode attempts recorded.")

    # ------- Distance checks + flagging; collect which coords to reverse (if needed) -------
    flagged_mask = []
    for idx, row in work.iterrows():
        r_list = [x for x in str(work.at[idx, "Reason"]).split(", ") if x] if work.at[idx, "Reason"] else []
        u_ll = user_coords[idx]
        p_ll = parent_coords[idx]
        q_used = ""
        f_meta = {}

        # If both coords absent, skip distance
        if (u_ll is None) and (p_ll is None):
            flagged_mask.append(bool(r_list))
            continue

        # Fetch client lat/lon from cache (if we created the key earlier)
        key = (
            str(row.get(COL_CLIENT_ADDR) or "").strip(),
            str(row.get(COL_CLIENT_CITY) or "").strip(),
            str(DEFAULT_STATE or "").strip(),
            str(row.get(COL_CLIENT_ZIP) or "").strip(),
        )
        client_ll, debug_note, q_used, f_meta = fwd_cache.get(key, (None, "NO_GEOCODE", "", {}))

        # Store the debug info (even if no geocode)
        work.at[idx, "Forward geocode query"] = q_used or ""
        work.at[idx, "Forward matching_text"] = f_meta.get("matching_text", "") if isinstance(f_meta, dict) else ""

        if client_ll is None:
            if "No user signature" not in r_list or "No parent signature" not in r_list:
                # Only add if we actually intended to compute distance
                r_list.append("Unable to geocode client address")
            work.at[idx, "Reason"] = ", ".join(sorted(set(r_list)))
            flagged_mask.append(True)
            continue

        # store client LL
        work.at[idx, "Client lat,lon"] = f"{client_ll[0]:.6f}, {client_ll[1]:.6f}"

        # Distances (only for valid coords)
        if u_ll is not None:
            d_user = haversine_m(client_ll, u_ll)
            if d_user is not None and d_user > DISTANCE_METERS_THRESHOLD:
                r_list.append(f"User signature > {DISTANCE_FEET_THRESHOLD} ft from client")
        if p_ll is not None:
            d_parent = haversine_m(client_ll, p_ll)
            if d_parent is not None and d_parent > DISTANCE_METERS_THRESHOLD:
                r_list.append(f"Parent signature > {DISTANCE_FEET_THRESHOLD} ft from client")

        work.at[idx, "Reason"] = ", ".join(sorted(set(r_list)))
        flagged_mask.append(bool(r_list))

    # ------- Reverse geocode signatures (deduped) -------
    rows_to_reverse = work.index[flagged_mask] if reverse_only_flagged else work.index

    need_rev = set()
    for idx in rows_to_reverse:
        u_ll = user_coords[idx]
        p_ll = parent_coords[idx]
        if u_ll: need_rev.add(("user", round(u_ll[0], 6), round(u_ll[1], 6)))
        if p_ll: need_rev.add(("parent", round(p_ll[0], 6), round(p_ll[1], 6)))

    st.info(f"Reverse-geocoding signature coordinates: {len(need_rev)} (deduped, {'flagged rows only' if reverse_only_flagged else 'all rows'})")

    rev_cache = {}
    if MAPBOX_TOKEN and need_rev:
        prog = st.progress(0.0)
        total = len(need_rev)
        for i, key in enumerate(need_rev, 1):
            _, lat, lon = key
            addr = _mapbox_reverse(lat, lon, MAPBOX_TOKEN)
            rev_cache[key] = addr
            prog.progress(i/total)
            time.sleep(0.02)
    else:
        if not MAPBOX_TOKEN:
            st.warning("Skipping reverse geocoding — no Mapbox token provided.")

    # Fill address columns for selected rows
    for idx in rows_to_reverse:
        u_ll = user_coords[idx]
        p_ll = parent_coords[idx]
        if u_ll:
            key = ("user", round(u_ll[0], 6), round(u_ll[1], 6))
            work.at[idx, "User signature address"] = rev_cache.get(key, work.at[idx, "User signature address"])
        if p_ll:
            key = ("parent", round(p_ll[0], 6), round(p_ll[1], 6))
            work.at[idx, "Parent signature address"] = rev_cache.get(key, work.at[idx, "Parent signature address"])

    # ------- Output flagged rows only -------
    flagged = work[work["Reason"].astype(str).str.strip() != ""].copy()
    out_cols = [c for c in VISIBLE_COLS if c in flagged.columns]

    st.subheader("Flagged rows only")
    st.write(f"Flagged: **{len(flagged)}** of **{len(work)}** filtered sessions.")
    st.dataframe(flagged[out_cols], use_container_width=True)

    st.download_button(
        "Download FLAGGED rows (CSV)",
        data=flagged[out_cols].to_csv(index=False).encode("utf-8"),
        file_name="flagged_signature_proximity_mapbox_fast.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV/XLSX to begin.")
