import streamlit as st
from twilio.rest import Client
import time

# Twilio credentials from secrets
TWILIO_SID = st.secrets["TWILIO_SID"]
TWILIO_AUTH_TOKEN = st.secrets["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = st.secrets["TWILIO_PHONE_NUMBER"]

# Initialize client
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

st.title("📲 Mass SMS Sender (Manual Entry + Delivery Tracking)")

# User inputs
numbers_text = st.text_area("Enter phone numbers (E.164 format, one per line)", height=200)
message = st.text_area("Enter message to send", height=150)

# Send button
if st.button("Send Messages"):
    if not numbers_text.strip() or not message.strip():
        st.warning("Please enter both phone numbers and a message.")
    else:
        phone_list = [
            line.strip() for line in numbers_text.strip().splitlines()
            if line.strip()
        ]

        sent_count = 0
        failed_numbers = []

        with st.spinner("Sending messages and checking delivery..."):
            for number in phone_list:
                try:
                    msg = client.messages.create(
                        body=message,
                        from_=TWILIO_PHONE_NUMBER,
                        to=number
                    )
                    # Wait a moment for delivery status to update
                    time.sleep(2)
                    status = client.messages(msg.sid).fetch().status
                    if status in ['delivered', 'sent', 'queued']:
                        sent_count += 1
                    else:
                        failed_numbers.append((number, status))
                except Exception as e:
                    failed_numbers.append((number, str(e)))

        st.success(f"✅ {sent_count} messages sent successfully.")
        if failed_numbers:
            st.error(f"❌ {len(failed_numbers)} messages failed.")
            for num, reason in failed_numbers:
                st.write(f"{num}: {reason}")