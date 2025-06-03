import streamlit as st
import pandas as pd
from twilio.rest import Client

# ---- Twilio Config (set your credentials here or use Streamlit Secrets) ----
TWILIO_SID = st.secrets.get("TWILIO_SID", "")
TWILIO_AUTH_TOKEN = st.secrets.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = st.secrets.get("TWILIO_PHONE_NUMBER", "")

st.title("📲 Mass SMS Sender via Twilio")

# ---- Upload CSV with phone numbers ----
uploaded_file = st.file_uploader("Upload CSV with 'phone' column", type=['csv'])

# ---- Message input ----
message = st.text_area("Enter the message to send", height=100)

# ---- Send button ----
if st.button("Send Messages"):
    if not all([TWILIO_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        st.error("Twilio credentials are missing. Set them in Streamlit secrets.")
    elif uploaded_file is None:
        st.warning("Please upload a CSV file with phone numbers.")
    elif not message.strip():
        st.warning("Please enter a message.")
    else:
        df = pd.read_csv(uploaded_file)

        if "phone" not in df.columns:
            st.error("CSV must contain a 'phone' column.")
        else:
            client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
            success_count = 0
            error_count = 0

            with st.spinner("Sending messages..."):
                for number in df["phone"]:
                    try:
                        client.messages.create(
                            body=message,
                            from_=TWILIO_PHONE_NUMBER,
                            to=str(number)
                        )
                        success_count += 1
                    except Exception as e:
                        error_count += 1
                        st.error(f"Failed to send to {number}: {e}")

            st.success(f"Messages sent successfully: {success_count}")
            if error_count:
                st.warning(f"Failed messages: {error_count}")