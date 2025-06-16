import streamlit as st
import pandas as pd
import re
from datetime import datetime
import os

st.title("Medicare Remittance Parser")

uploaded_file = st.file_uploader("Upload Remittance TXT File", type=["txt"])

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    base_filename = os.path.splitext(uploaded_file.name)[0]

    # Split each patient section
    patients = re.split(r"_+\n", content.strip())
    all_rows = []

    for patient in patients:
        name = re.search(r"NAME:(.+)", patient)
        hic = re.search(r"HIC:(.+)", patient)
        acnt = re.search(r"ACNT:(.+)", patient)
        icn = re.search(r"ICN:(\S+)", patient)

        name = name.group(1).strip() if name else ""
        hic = hic.group(1).strip() if hic else ""
        acnt = acnt.group(1).strip() if acnt else ""
        icn = icn.group(1).strip() if icn else ""

        service_lines = re.findall(
            r"(\d{10})\s+\d{4}\s+(\d{6})\n"  # NPI, service date
            r"(\d{5})\n"                    # billing code
            r"([\d.]+)\n"                  # units
            r"[\d.]+\n"                    # billed
            r"[\d.]+\n"                    # allowed
            r"[\d.]+\n"                    # deductible
            r"[\d.]+\n"                    # coins
            r"([\d.]+)",                   # paid
            patient
        )

        for provider, yymmdd, code, units, paid in service_lines:
            try:
                month = yymmdd[:2]
                day = yymmdd[2:4]
                year = "20" + yymmdd[4:]
                serv_date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").date()
            except ValueError:
                serv_date = ""

            all_rows.append({
                "NAME": name,
                "HIC": hic,
                "ACNT": acnt,
                "ICN": icn,
                "RENDERING PROVIDER": provider,
                "serv date": serv_date,
                "billingcode": code,
                "prov pd": float(paid),
                "units": int(float(units))
            })

    # Create DataFrame and Excel output
    df = pd.DataFrame(all_rows)
    output_filename = f"{base_filename}.xlsx"
    df.to_excel(output_filename, index=False)

    st.success(f"✅ File processed successfully: {output_filename}")

    with open(output_filename, "rb") as f:
        st.download_button("📥 Download Excel File", f, file_name=output_filename)