import streamlit as st
import pandas as pd

st.set_page_config(page_title="BT & Supervision Hour Calculator", layout="wide")
st.title("🧮 BT, Supervision & Parent Training Hour Calculator")

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Required columns
    required_columns = ['Client Name', 'Rendering Provider', 'Service Name', 'Appt. Date', 'Billing Hours', 'Completed']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Missing one or more required columns: {required_columns}")
        st.stop()

    # Filter completed sessions
    df = df[df['Completed'].str.strip().str.lower() == 'yes']

    # ------------------------------------
    # 1️⃣ Client-Level Summary
    # ------------------------------------
    bt_df = df[df['Service Name'].str.contains('BT', case=False)]
    sup_df = df[df['Service Name'].str.contains('Supervision', case=False)]

    bt_summary = bt_df.groupby(['Client Name', 'Rendering Provider'])['Billing Hours'].sum().reset_index()
    bt_summary.rename(columns={'Billing Hours': 'Total BT Hours'}, inplace=True)

    sup_summary = sup_df.groupby(['Client Name', 'Rendering Provider'])['Billing Hours'].sum().reset_index()
    sup_summary.rename(columns={'Billing Hours': 'Total Supervision Hours'}, inplace=True)

    summary = pd.merge(bt_summary, sup_summary,
                       on=['Client Name', 'Rendering Provider'],
                       how='outer').fillna(0)

    summary['Required Supervision (5%)'] = summary['Total BT Hours'] * 0.05
    summary[['Total BT Hours', 'Total Supervision Hours', 'Required Supervision (5%)']] = \
        summary[['Total BT Hours', 'Total Supervision Hours', 'Required Supervision (5%)']].round(2)

    # ------------------------------------
    # 2️⃣ Supervision Totals per LBA
    # ------------------------------------
    provider_summary = sup_df.groupby('Rendering Provider')['Billing Hours'].sum().reset_index()
    provider_summary.rename(columns={'Billing Hours': 'Total Supervision Hours'}, inplace=True)
    provider_summary['Total Supervision Hours'] = provider_summary['Total Supervision Hours'].round(2)

    # ------------------------------------
    # 3️⃣ Parent Training Summary
    # ------------------------------------
    parent_df = df[df['Service Name'].str.contains('Parent Training', case=False)]
    parent_summary = parent_df.groupby(['Client Name', 'Rendering Provider'])['Billing Hours'].sum().reset_index()
    parent_summary.rename(columns={'Billing Hours': 'Parent Training Hours'}, inplace=True)
    parent_summary['Parent Training Hours'] = parent_summary['Parent Training Hours'].round(2)

    # ------------------------------------
    # Display all 3 summaries
    # ------------------------------------
    st.success("✅ Summary generated successfully!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Client-Level BT & Supervision Summary")
        st.dataframe(summary, use_container_width=True)
        csv1 = summary.to_csv(index=False)
        st.download_button("📥 Download Client Summary CSV", csv1, "bt_client_summary.csv", "text/csv")

    with col2:
        st.subheader("👩‍⚕️ LBA Supervision Totals")
        st.dataframe(provider_summary, use_container_width=True)
        csv2 = provider_summary.to_csv(index=False)
        st.download_button("📥 Download LBA Summary CSV", csv2, "lba_supervision_summary.csv", "text/csv")

    st.subheader("👨‍👩‍👧 Parent Training Sessions")
    st.dataframe(parent_summary, use_container_width=True)
    csv3 = parent_summary.to_csv(index=False)
    st.download_button("📥 Download Parent Training CSV", csv3, "parent_training_summary.csv", "text/csv")

else:
    st.info("Upload a file with: Client Name, Rendering Provider, Service Name, Appt. Date, Billing Hours, Completed.")
