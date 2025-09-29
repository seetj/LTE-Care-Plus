import streamlit as st
import pandas as pd
from io import BytesIO

st.title("Conflict Checker: Supervision & Assessment Rules")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Normalize column names
    df.columns = [col.strip() for col in df.columns]

    required_cols = {'Appt. Date', 'Appt. Start Time', 'Appt. End Time', 'Client Name', 'Service Name', 'Staff Name'}
    if not required_cols.issubset(df.columns):
        st.error(f"Missing required columns: {required_cols - set(df.columns)}")
        st.stop()

    # Create datetime fields
    df['Start'] = pd.to_datetime(df['Appt. Date'].astype(str) + ' ' + df['Appt. Start Time'].astype(str), errors='coerce')
    df['End'] = pd.to_datetime(df['Appt. Date'].astype(str) + ' ' + df['Appt. End Time'].astype(str), errors='coerce')
    df = df.dropna(subset=['Start', 'End'])

    # Prepare additional fields
    df['Conflict'] = False
    df['Service Name Lower'] = df['Service Name'].fillna('').str.lower()

    # Subsets
    bt_df = df[df['Service Name Lower'].str.contains("direct service bt")]
    superv_df = df[df['Service Name Lower'] == 'supervision']
    assessment_df = df[df['Service Name Lower'] == 'assessment']

    conflict_rows = []

    def append_conflict(row, conflict_type):
        row_copy = row.copy()
        row_copy['Conflict Type'] = conflict_type
        conflict_rows.append(row_copy)

    ### Rule 1: Supervision must be inside BT session
    for idx, row in superv_df.iterrows():
        same_day_bt = bt_df[(bt_df['Client Name'] == row['Client Name']) & (bt_df['Appt. Date'] == row['Appt. Date'])]
        is_covered = any((bt['Start'] <= row['Start']) and (bt['End'] >= row['End']) for _, bt in same_day_bt.iterrows())

        if not is_covered:
            df.at[idx, 'Conflict'] = True
            append_conflict(row, 'Supervision Outside BT')
            for _, bt in same_day_bt.iterrows():
                append_conflict(bt, 'Reference BT Row')

    ### Rule 2: Assessment must not follow any BT session on same day
    for idx, row in assessment_df.iterrows():
        earlier_bt = bt_df[
            (bt_df['Client Name'] == row['Client Name']) &
            (bt_df['Appt. Date'] == row['Appt. Date']) &
            (bt_df['Start'] < row['Start'])
        ]
        if not earlier_bt.empty:
            df.at[idx, 'Conflict'] = True
            append_conflict(row, 'BT before Assessment')
            for _, bt in earlier_bt.iterrows():
                append_conflict(bt, 'Reference BT Row (before Assessment)')

    ### Rule 3: Overlapping Supervision by same staff
    for (staff, date), group in superv_df.groupby(['Staff Name', 'Appt. Date']):
        sorted_group = group.sort_values('Start').reset_index(drop=True)
        for i in range(len(sorted_group)):
            for j in range(i + 1, len(sorted_group)):
                r1 = sorted_group.iloc[i]
                r2 = sorted_group.iloc[j]
                if r1['End'] > r2['Start'] and r1['Start'] < r2['End']:
                    for r in [r1, r2]:
                        df.at[r.name, 'Conflict'] = True
                        append_conflict(r, 'Overlapping Supervision (Same Staff)')

    ### Rule 4: First 4 assessments must be completed before any BT
    for client, group in df.groupby('Client Name'):
        bt_days = set(bt_df[bt_df['Client Name'] == client]['Appt. Date'])
        assessment_same_client = assessment_df[assessment_df['Client Name'] == client]
        
        for idx, row in assessment_same_client.iterrows():
            if row['Appt. Date'] in bt_days:
                append_conflict(row, 'Assessment and BT on Same Day')
                # Add all BT rows on that same day for reference
                bt_same_day = bt_df[
                    (bt_df['Client Name'] == client) &
                    (bt_df['Appt. Date'] == row['Appt. Date'])
                ]
                for _, bt_row in bt_same_day.iterrows():
                    append_conflict(bt_row, 'Reference BT on Same Day')

    # Combine conflicts
    if conflict_rows:
        conflict_df = pd.DataFrame(conflict_rows)
    else:
        conflict_df = pd.DataFrame()

    # Display Results
    st.subheader("Detected Conflicts")
    if conflict_df.empty:
        st.success("✅ No conflicts found.")
    else:
        st.dataframe(conflict_df)

        def to_excel(dframe):
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine='openpyxl') as writer:
                dframe.to_excel(writer, index=False)
            bio.seek(0)
            return bio

        st.download_button(
            "Download Conflict Rows",
            to_excel(conflict_df),
            "conflict_rows.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
