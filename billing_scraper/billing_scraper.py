import streamlit as st
import pandas as pd
import re

# ---------- Amount helpers ----------

def normalize_amount(s: str) -> str:
    """Normalize amounts like -12.34, (12.34), $1,234.56 into plain strings."""
    if not s:
        return ''
    s = s.strip().replace(',', '').replace('$', '')
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    return s

# Non-capturing internals; avoids extra capture groups
AMT_NEG = r'(?:-?\(?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?)'  # accepts negatives and parentheses
AMT_POS = r'(?:\d+\.\d+)'                                  # original: positive decimals only

# ---------- Compiled regexes (NEG first, then POS fallback) ----------

HEADER_RE_NEG = re.compile(
    rf'(?P<check>\d{{10,20}})\s+(?P<patient_id>\S+)\s+'
    rf'(?P<name>[A-Z\'\-]+,[A-Z\'\-]+(?:\s+[A-Z])?)\s+'
    rf'(?P<charge_amt>{AMT_NEG})\s+(?P<payment_amt>{AMT_NEG})\s+'
    rf'(?P<accnt>[A-Z0-9]+)\s+(?P<status>PROCESSED AS (?:PRIMARY|SECONDARY)|DENIED|REJECTED|OTHER)'
    rf'(?:\s+(?P<payer>[A-Z]+))?',
    re.IGNORECASE
)

HEADER_RE_POS = re.compile(
    rf'(?P<check>\d{{10,20}})\s+(?P<patient_id>\S+)\s+'
    rf'(?P<name>[A-Z\'\-]+,[A-Z\'\-]+(?:\s+[A-Z])?)\s+'
    rf'(?P<charge_amt>{AMT_POS})\s+(?P<payment_amt>{AMT_POS})\s+'
    rf'(?P<accnt>[A-Z0-9]+)\s+(?P<status>PROCESSED AS (?:PRIMARY|SECONDARY)|DENIED|REJECTED|OTHER)'
    rf'(?:\s+(?P<payer>[A-Z]+))?',
    re.IGNORECASE
)

LINE_RE_NEG = re.compile(
    rf'(?P<svc_date>\d{{2}}/\d{{2}}/\d{{4}})\s+(?P<cpt>\d+)\s+'
    rf'(?P<line_charge>{AMT_NEG})\s+(?P<line_payment>{AMT_NEG})\s+(?P<total_adj>{AMT_NEG})?\s+(?P<remarks>.*)'
)

LINE_RE_POS = re.compile(
    r'(?P<svc_date>\d{2}/\d{2}/\d{4})\s+(?P<cpt>\d+)\s+'
    r'(?P<line_charge>[\d.]+)\s+(?P<line_payment>[\d.]+)\s+(?P<total_adj>[\d.]+)?\s+(?P<remarks>.*)'
)


def parse_billing_text_file(text: str, debug=False, allow_negatives=True):
    text = text.replace('\xa0', ' ')  # Normalize non-breaking spaces
    entries = re.split(r'\n-{152,}\n', text)

    records = []
    skipped_entries = []

    for idx, entry in enumerate(entries):
        entry = entry.strip()
        lines = entry.splitlines()
        if not lines:
            continue

        # remove empty lines
        lines = [line for line in lines if line.strip()]
        # drop leading "Check#"
        if lines and lines[0].strip().startswith("Check#"):
            lines = lines[1:]

        entry = "\n".join(lines)
        entry_flat = re.sub(r'\s+', ' ', entry)

        # ----- HEADER: try NEG matcher, then POS fallback -----
        header_match = (HEADER_RE_NEG.search(entry_flat) if allow_negatives else None) or HEADER_RE_POS.search(entry_flat)
        used_header_pat = "NEG" if (allow_negatives and header_match and HEADER_RE_NEG.search(entry_flat)) else "POS"

        if not header_match:
            skipped_entries.append(entry[:500])
            if debug:
                st.write(f"⚠️ Skipped entry #{idx} (no header match). Tried {'NEG then POS' if allow_negatives else 'POS only'}.")
                st.code(entry[:500], language="text")
            continue

        header_info = header_match.groupdict()
        # Default payer if missing
        header_info["payer"] = header_info.get("payer") or "NYSDOH"
        # normalize header amounts
        header_info["charge_amt"] = normalize_amount(header_info.get("charge_amt"))
        header_info["payment_amt"] = normalize_amount(header_info.get("payment_amt"))
        header_info["status"] = (header_info.get("status") or '').strip()

        # ----- payer details -----
        payer_details_parts = []
        payer_address_match = re.search(
            r'(OFFICE OF HEALTH INSURANCE PROGRAM).*?(CORNING TOWER, EMPIRE STATE PLAZA).*?'
            r'(ALBANY,NY \d+).*?(Tax ID: \S+)', entry, re.DOTALL)
        payer_claim_number_match = re.search(r'Payer Claim Control Number: (\d+)', entry)

        if payer_address_match:
            payer_details_parts.extend(payer_address_match.groups())
        if payer_claim_number_match:
            payer_details_parts.append(f'Payer Claim Control Number: {payer_claim_number_match.group(1)}')

        header_info["payer_details"] = " | ".join(payer_details_parts)

        # ----- claim period -----
        claim_period_match = re.search(
            r'Claim Statement Period:\s+(\d{2}/\d{2}/\d{4}) - (\d{2}/\d{2}/\d{4})', entry)
        header_info["claim_start"] = claim_period_match.group(1) if claim_period_match else ''
        header_info["claim_end"] = claim_period_match.group(2) if claim_period_match else ''

        # ----- LINE ITEMS -----
        raw_lines = re.split(r'Line Item:', entry)
        if len(raw_lines) <= 1:
            if debug:
                st.write(f"⚠️ Entry #{idx} had no line items (header matched with {used_header_pat}).")
            continue

        for block in raw_lines[1:]:
            block = block.strip()

            # Try NEG first (if enabled), then POS fallback
            svc_match = (LINE_RE_NEG.search(block) if allow_negatives else None) or LINE_RE_POS.search(block)
            used_line_pat = "NEG" if (allow_negatives and svc_match and LINE_RE_NEG.search(block)) else "POS"

            if not svc_match:
                if debug:
                    st.write(f"⚠️ Entry #{idx} line block skipped (no svc match). Tried {'NEG then POS' if allow_negatives else 'POS only'}.")
                    st.code(block[:300], language="text")
                continue

            g = svc_match.groupdict()
            svc_date = g.get("svc_date", "").strip()
            cpt = g.get("cpt", "").strip()
            charge_amt = normalize_amount(g.get("line_charge"))
            payment_amt = normalize_amount(g.get("line_payment"))
            total_adj_amt = normalize_amount(g.get("total_adj") or '')
            remarks = (g.get("remarks") or '').strip()

            # parse adjustment group/reason (kept as-is)
            adj_group = ''
            adj_amt_val = ''
            reason = ''
            if "Adjustment Group" in block:
                adj_lines = block.splitlines()
                for i, line in enumerate(adj_lines):
                    if "Adjustment Group" in line and i + 1 < len(adj_lines):
                        adj_values = adj_lines[i + 1].strip()
                        adj_parts = re.split(r'\s{2,}', adj_values)
                        if len(adj_parts) >= 3:
                            adj_group = adj_parts[0].strip()
                            adj_amt_val = adj_parts[1].strip()
                            reason = adj_parts[2].strip()

            record = {
                **header_info,
                "svc_date": svc_date,
                "cpt": cpt,
                "line_charge_amt": charge_amt,
                "line_payment_amt": payment_amt,
                "total_adj_amt": total_adj_amt,
                "remarks": remarks,
                "adj_group": adj_group,
                "adj_group_amt": normalize_amount(adj_amt_val),
                "reason": reason
            }

            if debug:
                record["_matched_header"] = used_header_pat
                record["_matched_line"] = used_line_pat

            records.append(record)

    df = pd.DataFrame(records)

    # Optional: convert to numeric (won't error; bad values -> NaN)
    for col in ["charge_amt", "payment_amt",
                "line_charge_amt", "line_payment_amt",
                "total_adj_amt", "adj_group_amt"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, skipped_entries


# ---------------- Streamlit UI ----------------

st.title("Billing Text File Parser")

uploaded_file = st.file_uploader("Upload Billing Text File", type=["txt"])
debug_mode = st.checkbox("🔍 Debug Mode (show skipped/mismatched entries)")
allow_neg = st.checkbox("➖ Accept negative / (parenthesized) amounts", value=True)

if uploaded_file is not None:
    try:
        raw_text = uploaded_file.read().decode("utf-8").replace('\r\n', '\n')
        df, skipped = parse_billing_text_file(raw_text, debug=debug_mode, allow_negatives=allow_neg)

        if df.empty:
            st.warning("No valid records found.")
        else:
            st.success(f"✅ Parsed {len(df)} rows. ❗ Skipped {len(skipped)} entries.")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "parsed_billing_data.csv", "text/csv")

        if skipped:
            if not debug_mode:
                with st.expander("⚠️ View Skipped Entries (Missing Headers)"):
                    for i, snippet in enumerate(skipped[:10]):
                        st.code(snippet, language="text")
                    if len(skipped) > 10:
                        st.write(f"... and {len(skipped) - 10} more entries were skipped.")

            # Download all skipped entries as text file
            skipped_text = "\n\n---\n\n".join(skipped)
            st.download_button(
                "📥 Download Skipped Entries",
                skipped_text.encode("utf-8"),
                "skipped_entries.txt",
                "text/plain"
            )

    except Exception as e:
        st.error(f"❌ Failed to parse file: {e}")
