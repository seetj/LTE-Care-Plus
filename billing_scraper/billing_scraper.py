import streamlit as st
import pandas as pd
import re

def parse_billing_text_file(text: str):
    text = text.replace('\xa0', ' ')  # Normalize non-breaking spaces
    entries = re.split(r'\n-{152,}\n', text)
  # More flexible entry splitting
    records = []
    skipped_entries = []

    for entry in entries:
        entry = entry.strip()
        lines = entry.splitlines()
        if not lines:
            continue

        lines = [line for line in lines if line.strip()]
        if lines and lines[0].strip().startswith("Check#"):
            lines = lines[1:]

        entry = "\n".join(lines)
        entry_flat = re.sub(r'\s+', ' ', entry)

        header_match = re.search(
    r'(?P<check>\d{10,20})\s+(?P<patient_id>\S+)\s+'
    r'(?P<name>[A-Z\'\-]+,[A-Z\'\-]+(?:\s+[A-Z])?)\s+'
    r'(?P<charge_amt>\d+\.\d+)\s+(?P<payment_amt>\d+\.\d+)\s+'
    r'(?P<accnt>[A-Z0-9]+)\s+(?P<status>PROCESSED AS (?:PRIMARY|SECONDARY)|DENIED|REJECTED)'
    r'(?:\s+(?P<payer>[A-Z]+))?',
    entry_flat,
    re.IGNORECASE
)
        
        
        if not header_match:
            skipped_entries.append(entry[:500])  # Save a snippet for review
            continue


        header_info = header_match.groupdict()
        header_info["payer"] = "NYSDOH"

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

        claim_period_match = re.search(
            r'Claim Statement Period:\s+(\d{2}/\d{2}/\d{4}) - (\d{2}/\d{2}/\d{4})', entry)
        header_info["claim_start"] = claim_period_match.group(1) if claim_period_match else ''
        header_info["claim_end"] = claim_period_match.group(2) if claim_period_match else ''

        raw_lines = re.split(r'Line Item:', entry)
        if len(raw_lines) <= 1:
            continue

        for block in raw_lines[1:]:
            block = block.strip()
            svc_match = re.search(
                r'(\d{2}/\d{2}/\d{4})\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)?\s+(.*)',
                block
            )

            if not svc_match:
                continue

            svc_date, cpt, charge_amt, payment_amt, total_adj_amt, remarks = svc_match.groups()

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

            record = header_info.copy()
            record.update({
                "svc_date": svc_date.strip(),
                "cpt": cpt.strip(),
                "line_charge_amt": charge_amt.strip(),
                "line_payment_amt": payment_amt.strip(),
                "total_adj_amt": (total_adj_amt or '').strip(),
                "remarks": remarks.strip(),
                "adj_group": adj_group,
                "adj_group_amt": adj_amt_val,
                "reason": reason
            })
            records.append(record)

    df = pd.DataFrame(records)
    return df, skipped_entries

# ---------------- Streamlit UI ----------------

st.title("Billing Text File Parser")

uploaded_file = st.file_uploader("Upload Billing Text File", type=["txt"])

if uploaded_file is not None:
    try:
        raw_text = uploaded_file.read().decode("utf-8").replace('\r\n', '\n')
        df, skipped = parse_billing_text_file(raw_text)

        if df.empty:
            st.warning("No valid records found.")
        else:
            st.success(f"✅ Parsed {len(df)} rows. ❗ Skipped {len(skipped)} entries.")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "parsed_billing_data.csv", "text/csv")

        if skipped:
            with st.expander("⚠️ View Skipped Entries (Missing Headers)"):
                for i, snippet in enumerate(skipped[:10]):
                    st.code(snippet, language="text")
                if len(skipped) > 10:
                    st.write(f"... and {len(skipped) - 10} more entries were skipped.")

    except Exception as e:
        st.error(f"❌ Failed to parse file: {e}")
