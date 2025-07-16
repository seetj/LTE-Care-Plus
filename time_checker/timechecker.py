import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="City to Borough Mapper", layout="wide")
st.title("🏙️ City to Borough/Area Mapper")

uploaded_file = st.file_uploader("Upload CSV or Excel File with a City Column", type=["csv", "xlsx"])

def map_city_to_borough(city):
    if not isinstance(city, str):
        return ""

    city = city.strip().upper()
    print(city)

    queens_cities = {"QUEENS","MASPETH","LONG IS CITY", "LONG ISLAND CITY","LITTLE NECK","KEW GARDENS","OZONE PARK","BAYSIDE","WHITESTONE","NEW HYDE PARK", "SOUTH RICHMOND HILL","FRESH MEADOWS","MANHASSET","SPRINGFIELD GARDEN", "ELMHURST", "WOODSIDE", "QUEENS VILLAGE", "FLUSHING", "JAMAICA", "ASTORIA", "FAR ROCKAWAY", "REGO PARK", "RIDGEWOOD", "CORONA"}
    brooklyn_cities = {"BROOKLYN", "BKLYN"}
    bronx_cities = {"BRONX"}
    manhattan_cities = {"NEW YORK", "MANHATTAN", "NYC"}
    staten_island_cities = {"STATEN ISLAND"}
    long_island_cities = {"LONG ISLAND","RONKONKOMA","EAST MEADOW", "HEMPSTEAD", "VALLEY STREAM", "FREEPORT", "MASSAPEQUA", "GARDEN CITY", "NASSAU", "SUFFOLK"}
    westchester_cities = {"YONKERS","JERICHO","NESCONSET","TUCKAHOE", "WHITE PLAINS", "MOUNT VERNON", "NEW ROCHELLE", "WESTCHESTER"}

    if city in queens_cities:
        return "Queens"
    if city in brooklyn_cities:
        return "Brooklyn"
    if city in bronx_cities:
        return "Bronx"
    if city in manhattan_cities:
        return "Manhattan"
    if city in staten_island_cities:
        return "Staten Island"
    if city in long_island_cities:
        return "Long Island"
    if city in westchester_cities:
        return "Westchester"

    return "Other"

def convert_df_to_download(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    return output

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        city_col = st.selectbox("Select the City Column", df.columns)

        df["borough/area"] = df[city_col].apply(map_city_to_borough)

        st.success("✅ Mapped borough/area successfully.")
        st.dataframe(df.head(20))

        download_data = convert_df_to_download(df)
        st.download_button("📥 Download Updated File", data=download_data, file_name="city_mapped_addresses.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👆 Upload a file to begin.")
