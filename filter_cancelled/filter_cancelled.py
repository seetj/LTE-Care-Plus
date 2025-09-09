# remove_matches_app.py
import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Remove Matching Rows", page_icon="🧹", layout="wide")
st.title("🧹 Remove Matching Rows Between Two Sheets")

st.markdown("""
Upload your **Main sheet** (rows to keep) and a **Blocklist sheet** (rows to remove if they match).
By default, rows match on **Appt. Start**, **Appt. End**, and **Client Name**.
""")

@st.cache_data(show_spinner=False)
def load_table(file, sheet=None):
    name = getattr(file, "name", "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        # Excel: allow choosing sheet
        df = pd.read_excel(file, sheet_name=sheet)
    return df

def normalize_for_match(df, cols, datetime_cols=None):
    df2 = df.copy()

    # Trim spaces and unify case for string columns
    for c in cols:
        if c in df2.columns:
            if pd.api.types.is_string_dtype(df2[c]):
                df2[c] = df2[c].fillna("").astype(str).str.strip()
            else:
                # still coerce to string for matching safety
                df2[c] = df2[c].astype(str).fillna("").str.strip()

    # Attempt to parse datetime columns consistently (optional)
    if datetime_cols:
        for c in datetime_cols:
            if c in df2.columns:
                df2[c] = pd.to_datetime(df2[c], errors="coerce", infer_datetime_format=True)
    return df2

def downloadable_excel(df, filename="result.xlsx"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned")
    buf.seek(0)
    st.download_button("📥 Download cleaned Excel", data=buf, file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.sidebar:
    st.header("Settings")
    default_keys = ["Appt. Start", "Appt. End", "Client Name"]
    st.caption("You can change the match columns after uploading files.")

col1, col2 = st.columns(2)

with col1:
    main_file = st.file_uploader("Main sheet (rows to keep)", type=["csv", "xlsx", "xls"], key="main")
    main_sheet_name = None
    if main_file and main_file.name.lower().endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(main_file)
            main_sheet_name = st.selectbox("Select sheet (Main)", xls.sheet_names, index=0)
        except Exception:
            main_sheet_name = None

with col2:
    block_file = st.file_uploader("Blocklist sheet (rows to remove)", type=["csv", "xlsx", "xls"], key="block")
    block_sheet_name = None
    if block_file and block_file.name.lower().endswith((".xlsx", ".xls")):
        try:
            xls2 = pd.ExcelFile(block_file)
            block_sheet_name = st.selectbox("Select sheet (Blocklist)", xls2.sheet_names, index=0)
        except Exception:
            block_sheet_name = None

if main_file and block_file:
    try:
        df_main_raw = load_table(main_file, sheet=main_sheet_name)
        df_block_raw = load_table(block_file, sheet=block_sheet_name)

        st.subheader("Column Mapping")
        st.caption("Pick the columns to match on. Defaults shown below.")

        # Build choices based on current columns; preselect defaults if they exist
        def preselect(cols, target):
            # choose target if present, else leave empty
            return target if target in cols else None

        main_cols = list(df_main_raw.columns)
        block_cols = list(df_block_raw.columns)

        # Select match columns for BOTH dataframes (let users point disparate header names to logical keys)
        c1, c2, c3 = st.columns(3)

        with c1:
            main_appt_start = st.selectbox("Main: Appt. Start", main_cols, index=main_cols.index(preselect(main_cols, "Appt. Start")) if preselect(main_cols, "Appt. Start") else 0)
            block_appt_start = st.selectbox("Blocklist: Appt. Start", block_cols, index=block_cols.index(preselect(block_cols, "Appt. Start")) if preselect(block_cols, "Appt. Start") else 0)

        with c2:
            main_appt_end = st.selectbox("Main: Appt. End", main_cols, index=main_cols.index(preselect(main_cols, "Appt. End")) if preselect(main_cols, "Appt. End") else 0)
            block_appt_end = st.selectbox("Blocklist: Appt. End", block_cols, index=block_cols.index(preselect(block_cols, "Appt. End")) if preselect(block_cols, "Appt. End") else 0)

        with c3:
            main_client = st.selectbox("Main: Client Name", main_cols, index=main_cols.index(preselect(main_cols, "Client Name")) if preselect(main_cols, "Client Name") else 0)
            block_client = st.selectbox("Blocklist: Client Name", block_cols, index=block_cols.index(preselect(block_cols, "Client Name")) if preselect(block_cols, "Client Name") else 0)

        # Options
        st.markdown("---")
        o1, o2 = st.columns(2)
        with o1:
            case_insensitive = st.checkbox("Case-insensitive match for names", value=True)
        with o2:
            parse_datetimes = st.checkbox("Parse Appt. Start/End as datetimes", value=True)

        # Normalize copies for matching
        main_keys = [main_appt_start, main_appt_end, main_client]
        block_keys = [block_appt_start, block_appt_end, block_client]

        # Build normalized working frames with aligned temporary key names
        df_main = df_main_raw.copy()
        df_block = df_block_raw.copy()

        # Create aligned temp columns
        aligned = [("ApptStart_tmp", main_appt_start, block_appt_start),
                   ("ApptEnd_tmp",   main_appt_end,   block_appt_end),
                   ("Client_tmp",    main_client,     block_client)]

        for tmp, mcol, bcol in aligned:
            df_main[tmp]  = df_main[mcol]
            df_block[tmp] = df_block[bcol]

        # Normalize
        datetime_cols = ["ApptStart_tmp", "ApptEnd_tmp"] if parse_datetimes else None
        df_main = normalize_for_match(df_main, ["ApptStart_tmp", "ApptEnd_tmp", "Client_tmp"], datetime_cols=datetime_cols)
        df_block = normalize_for_match(df_block, ["ApptStart_tmp", "ApptEnd_tmp", "Client_tmp"], datetime_cols=datetime_cols)

        # Case-insensitive name matching (apply .str.lower())
        if case_insensitive:
            df_main["Client_tmp"] = df_main["Client_tmp"].astype(str).str.lower()
            df_block["Client_tmp"] = df_block["Client_tmp"].astype(str).str.lower()

        # If datetimes, format to a canonical string to avoid tz/precision mismatches
        if parse_datetimes:
            for c in ["ApptStart_tmp", "ApptEnd_tmp"]:
                df_main[c]  = df_main[c].dt.strftime("%Y-%m-%d %H:%M:%S")
                df_block[c] = df_block[c].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Build composite keys (tuple-like strings) for fast set membership
        def key_series(d):
            return (
                d["ApptStart_tmp"].astype(str).fillna("") + "||" +
                d["ApptEnd_tmp"].astype(str).fillna("")   + "||" +
                d["Client_tmp"].astype(str).fillna("")
            )

        df_main["_match_key"]  = key_series(df_main)
        df_block["_match_key"] = key_series(df_block)

        block_keys_set = set(df_block["_match_key"].unique())
        df_main["__is_blocked"] = df_main["_match_key"].isin(block_keys_set)

        removed_count = int(df_main["__is_blocked"].sum())
        kept_count = int((~df_main["__is_blocked"]).sum())

        st.success(f"✅ Found **{removed_count}** matching rows to remove. **{kept_count}** rows will remain.")

        with st.expander("Preview rows to be removed", expanded=False):
            st.dataframe(df_main.loc[df_main["__is_blocked"]].head(50), use_container_width=True)

        with st.expander("Preview cleaned result (first 200 rows)", expanded=True):
            cleaned = df_main.loc[~df_main["__is_blocked"]].drop(columns=["ApptStart_tmp","ApptEnd_tmp","Client_tmp","_match_key","__is_blocked"], errors="ignore")
            st.dataframe(cleaned.head(200), use_container_width=True)

        # Download buttons
        downloadable_excel(cleaned, filename="cleaned_without_matches.xlsx")

        # Also offer CSV
        st.download_button(
            "📄 Download cleaned CSV",
            data=cleaned.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_without_matches.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.caption("Tip: If your input date/time formats vary, enable **Parse Appt. Start/End as datetimes** to standardize matching.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.info("Upload both the **Main** and **Blocklist** files to begin.")
