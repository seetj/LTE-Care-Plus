import re
import pandas as pd
from datetime import datetime

# Input/output filenames
input_file = "1.27-2.2.txt"
output_file = "medicare_remits_output.xlsx"

# Load full text
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Split by solid lines (if any)
patients = re.split(r"_+\n", content.strip())

all_rows = []

for patient in patients:
    # Extract header fields
    name = re.search(r"NAME:(.+)", patient)
    hic = re.search(r"HIC:(.+)", patient)
    acnt = re.search(r"ACNT:(.+)", patient)
    icn = re.search(r"ICN:(\S+)", patient)

    name = name.group(1).strip() if name else ""
    hic = hic.group(1).strip() if hic else ""
    acnt = acnt.group(1).strip() if acnt else ""
    icn = icn.group(1).strip() if icn else ""

    # Match service blocks
    service_lines = re.findall(
        r"(\d{10})\s+\d{4}\s+(\d{6})\n"  # NPI and 6-digit date
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
            "units": float(units)
        })

# Save to Excel
df = pd.DataFrame(all_rows)
df.to_excel(output_file, index=False)
print(f"✅ Successfully saved to {output_file}")
