import streamlit as st
import pandas as pd
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils import split_name, retry_on_stale, random_sleep, normalize_gender,split_city_state_zip
from element_ids import ElementIDS 
from datetime import date
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

# Setup logging
logging.basicConfig(filename='epaces_scraper.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

URL = "https://epaces.emedny.org/Login.aspx"

USERNAME = st.text_input("Username", "ALTE1")
PASSWORD = st.text_input("Password", "Lte123456", type="password")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    processed_items = []

    try:
        dataframe = pd.read_excel(uploaded_file)
        st.success("File successfully read!")
        dict_array = dataframe.to_dict(orient='records')

        options = uc.ChromeOptions()
        options.add_argument("--headless=new")         
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--no-sandbox")

        driver = uc.Chrome(options=options)
        driver.get(URL)

        retry_on_stale(lambda: driver.find_element(By.ID, "Username").send_keys(USERNAME))
        retry_on_stale(lambda: driver.find_element(By.ID, "Password").send_keys(PASSWORD))
        checkbox = retry_on_stale(lambda: driver.find_element(By.ID, "chkbxAgree"))
        if not checkbox.is_selected():
            retry_on_stale(lambda: checkbox.click())
        retry_on_stale(lambda: driver.find_element(By.ID, "btnAgreeLogin").click())

        random_sleep(2, 4)

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for idx, item in enumerate(tqdm(dict_array, desc="Processing", unit="item")):
            try:
                driver.get("https://epaces.emedny.org/home/LandingPSO.aspx?strAnn=1")
                random_sleep(4,5)
                retry_on_stale(lambda: driver.find_element(By.ID, ElementIDS.REQUEST_ID).click())
                random_sleep(4,5)
                retry_on_stale(lambda: driver.find_element(By.ID, ElementIDS.INPUT_CLIENT_ID).send_keys(item["Medicaid"]))
                random_sleep(5, 6)
                submit_button = retry_on_stale(lambda: driver.find_element(By.ID, ElementIDS.SUBMIT_BUTTON_ID))
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", submit_button)
                random_sleep(1, 2)
                retry_on_stale(lambda: submit_button.click())
                random_sleep(10,11)
                retry_on_stale(lambda: driver.find_element(By.ID, ElementIDS.RESPONSE_ID).click())
                random_sleep(1, 2)

                element_result = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, item["Medicaid"]))
                )
                retry_on_stale(lambda: element_result.click())
                random_sleep(2, 3)

                page_text = driver.page_source
                soup = BeautifulSoup(page_text, 'html.parser')
                # Set default values
                item['new Medicaid'] = "STR MEDICAID"
                item['new Code 95'] = "N"
                name = soup.find(id=ElementIDS.CLIENTNAME_ID).text
                item['First'],item['Middle'],item['Last'] = split_name(name)
                item['Gender'] = normalize_gender(soup.find(id=ElementIDS.CLIENT_GENDER_ID).text)
                item['DOB'] = soup.find(id=ElementIDS.DATE_OF_BIRTH_ID).text
                item['Address'] = soup.find(id=ElementIDS.ADDRESS_ID).text.strip().title()
                item['City'],item['State'],item['Zip Code'] = split_city_state_zip(soup.find(id=ElementIDS.CITY_STATE_ZIP_ID).text)

                for table in soup.find_all('table'):
                    if table.find('th', string="Exemption Code"):
                        for td in table.find_all('td'):
                            if td.text.strip() == "95":
                                item['new Code 95'] = "Y"

                processed_items.append(item)

                random_sleep(3,4)
                progress = (idx + 1) / len(dict_array)
                progress_bar.progress(progress)
                progress_text.text(f"Progress: {int(progress * 100)}%")

                logging.info(f"Processed {item['Medicaid']} successfully.")

            except Exception as item_error:
                logging.error(f"Error processing {item.get('Medicaid')}: {item_error}")

        st.success("Processing complete or interrupted. Exporting data...")

    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        st.error(f"Script stopped due to error: {e}")

    finally:
        # Export whatever has been processed so far
        if processed_items:
            df_done = pd.DataFrame(processed_items)
            today_date = date.today()
            output_file = f"{today_date} new cases.xlsx"
            df_done.to_excel(output_file, index=False)
            st.download_button(
                label="Download File",
                data=open(output_file, 'rb').read(),
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No data processed.")
        try:
            driver.quit()
        except:
            pass

        