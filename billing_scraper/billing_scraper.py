import streamlit as st
import pandas as pd
import re
import io

st.title("Billing Text File Parser")

uploaded_file = st.file_uploader("Upload Billing Text File", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    entries = text.strip().split('\n' + '-' * 100 + '\n')
    records = []

    for entry in entries:
        header_match = re.search(
            r'(?P<check>\d+)\s+(?P<patient_id>\S+)\s+(?P<name>[A-Z]+,[A-Z]+)\s+'
            r'(?P<charge_amt>\d+\.\d+)\s+(?P<payment_amt>\d+\.\d+)\s+'
            r'(?P<accnt>P\d+)\s+(?P<status>PROCESSED AS (?:PRIMARY|SECONDARY)|DENIED|REJECTED)\s+'
            r'(?P<payer>NYSDOH)', entry)

        if not header_match:
            continue

        header_info = header_match.groupdict()

        payer_address_match = re.search(
            r'(OFFICE OF HEALTH INSURANCE PROGRAM).*?(CORNING TOWER, EMPIRE STATE PLAZA).*?'
            r'(ALBANY,NY \d+).*?(Tax ID: \S+)', entry, re.DOTALL)
        payer_claim_number_match = re.search(r'Payer Claim Control Number: (\d+)', entry)
        payer_details_parts = []

        if payer_address_match:
            payer_details_parts.extend(payer_address_match.groups())
        if payer_claim_number_match:
            payer_details_parts.append(f'Payer Claim Control Number: {payer_claim_number_match.group(1)}')

        header_info['payer_details'] = " | ".join(payer_details_parts)

        claim_period_match = re.search(
            r'Claim Statement Period:\s+(\d{2}/\d{2}/\d{4}) - (\d{2}/\d{2}/\d{4})', entry)
        header_info['claim_start'] = claim_period_match.group(1) if claim_period_match else ''
        header_info['claim_end'] = claim_period_match.group(2) if claim_period_match else ''

        raw_lines = re.split(r'Line Item:', entry)[1:]
        for block in raw_lines:
            svc_match = re.search(
            r'(\d{2}/\d{2}/\d{4})\s+(\d{5})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)',
            block.strip())

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
                'svc_date': svc_date.strip(),
                'cpt': cpt.strip(),
                'line_charge_amt': charge_amt.strip(),
                'line_payment_amt': payment_amt.strip(),
                'total_adj_amt': total_adj_amt.strip(),
                'remarks': remarks.strip(),
                'adj_group': adj_group,
                'adj_group_amt': adj_amt_val,
                'reason': reason
            })
            records.append(record)

    df_result = pd.DataFrame(records)
    st.dataframe(df_result)

    # Download as Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_result.to_excel(writer, index=False, sheet_name='Parsed Data')
    output.seek(0)

    st.download_button(
        label="Download Excel File",
        data=output,
        file_name="parsed_billing_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
