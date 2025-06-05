import streamlit as st
import pandas as pd
import re

def parse_billing_text_file(text: str) -> pd.DataFrame:
    entries = re.split(r'\n-{50,}\n', text.strip())
    records = []

    for entry in entries:
        lines = entry.strip().splitlines()
        if lines and lines[0].strip().startswith("Check#"):
            lines = lines[1:]

        if not lines:
            continue

        # Extract check number, patient ID, and name from first line
        first_line = lines[0].strip()
        check_match = re.match(
            r'(?P<check>\d+)\s+(?P<patient_id>\S+)\s+(?P<name>[\w\'\-]+,[\w\'\-]+)',
            first_line
        )
        if not check_match:
            continue

        header_info = check_match.groupdict()
        header_info["payer"] = "NYSDOH"

        # Recombine and flatten the full entry for other details
        entry = "\n".join(lines)
        entry_flat = re.sub(r'\s+', ' ', entry)

        # Extract remaining header fields
        accnt_match = re.search(r'(P\d+)', entry_flat)
        status_match = re.search(r'(PROCESSED AS (?:PRIMARY|SECONDARY)|DENIED|REJECTED)', entry_flat)

        header_info["accnt"] = accnt_match.group(1) if accnt_match else ''
        header_info["status"] = status_match.group(1) if status_match else ''

        # Payer details
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

        # Claim statement period
        claim_period_match = re.search(
            r'Claim Statement Period:\s+(\d{2}/\d{2}/\d{4}) - (\d{2}/\d{2}/\d{4})', entry)
        header_info["claim_start"] = claim_period_match.group(1) if claim_period_match else ''
        header_info["claim_end"] = claim_period_match.group(2) if claim_period_match else ''

        # Parse line items
        raw_lines = re.split(r'Line Item:', entry)[1:]
        for block in raw_lines:
            svc_match = re.search(
                r'(\d{2}/\d{2}/\d{4})\s+(\d{5})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)',
                block.strip())
            if not svc_match:
                continue

            svc_date, cpt, charge_amt, payment_amt, total_adj_amt, remarks = svc_match.groups()

            # Adjustments
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
                "total_adj_amt": total_adj_amt.strip(),
                "remarks": remarks.strip(),
                "adj_group": adj_group,
                "adj_group_amt": adj_amt_val,
                "reason": reason
            })
            records.append(record)

    return pd.DataFrame(records)

st.title("📄 Billing Text File Parser")

uploaded_file = st.file_uploader("Upload Billing Text File (.txt)", type=["txt"])

if uploaded_file:
    try:
        text = uploaded_file.read().decode("utf-8").replace('\r\n', '\n')
        df = parse_billing_text_file(text)

        st.success(f"✅ Parsed {len(df)} service lines for {df['name'].nunique()} patient(s)")
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error parsing file: {e}")
