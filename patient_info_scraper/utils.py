import logging
import time
import random
from selenium.common.exceptions import StaleElementReferenceException


def split_name(full_name):
    full_name = full_name.strip().title()
    if "," not in full_name:
        return "", "", full_name

    last, first_middle = full_name.split(",", 1)
    parts = first_middle.strip().split()

    first_name = parts[0] if len(parts) > 0 else ""
    middle_name = parts[1] if len(parts) > 1 else ""

    return first_name, middle_name, last.strip()

def retry_on_stale(action, retries=3, delay=1):
    for attempt in range(1, retries + 1):
        try:
            result = action()
            if attempt > 1:
                logging.info(f"Success on retry {attempt}")
            return result
        except StaleElementReferenceException as e:
            logging.warning(f"StaleElementReferenceException on attempt {attempt}: {e}")
            time.sleep(delay)
    logging.error(f"Element stayed stale after {retries} retries")
    raise StaleElementReferenceException("Element stayed stale after retries")

def random_sleep(min_seconds, max_seconds):
    time.sleep(random.uniform(min_seconds, max_seconds))

def normalize_gender(gender_letter):
    gender = gender_letter.strip().upper()
    if gender == "M":
        return "Male"
    elif gender == "F":
        return "Female"
    return ""  # or "Unknown"

def split_city_state_zip(location_str):
    location_str = location_str.strip()
    city_part, state_zip_part = location_str.split(",", 1)
    city = city_part.strip().title()
    state_zip_parts = state_zip_part.strip().split()
    state = state_zip_parts[0]
    zip_code = state_zip_parts[1] if len(state_zip_parts) > 1 else ""
    return city,state,zip_code
