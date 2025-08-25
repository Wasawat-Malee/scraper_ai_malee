# scraper_ai_malee.py
# ดึงราคาหุ้น MALEE จากหน้า SET ด้วยแนวทาง AI (ไม่พึ่งพา CSS selector)
# ต้องมี GOOGLE_API_KEY อยู่ใน environment (GitHub Actions: ใช้ secrets)

import os
import json
import pytz
import mimetypes
from datetime import datetime

import google.generativeai as genai

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


SYMBOL = "MALEE"
URL = "https://www.set.or.th/th/market/product/stock/quote/MALEE/price"
SCREENSHOT_PATH = "set_malee.png"


# ---------- 1) ตั้งค่า Gemini ด้วย GOOGLE_API_KEY ----------
def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ไม่พบ GOOGLE_API_KEY ใน environment. "
            "ให้ตั้งค่าใน GitHub Actions เป็น secrets แล้ว export เป็น env ของ job"
        )
    genai.configure(api_key=api_key)


# ---------- 2) เรนเดอร์หน้าเว็บ (text + screenshot) ----------
def render_set_page(url: str, screenshot_path: str = SCREENSHOT_PATH) -> tuple[str, str]:
    # เตรียม Chrome headless
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,1200")

    # เผื่อกรณีใช้ setup-chrome action ซึ่งกำหนด CHROME_PATH
    chrome_path = os.getenv("CHROME_PATH")
    if chrome_path:
        options.binary_location = chrome_path

    # ใช้ webdriver-manager จัดการไดรเวอร์
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)

        # รอให้หน้าโหลดสมบูรณ์
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        page_text = driver.find_element(By.TAG_NAME, "body").text
        driver.save_screenshot(screenshot_path)
        return page_text, screenshot_path
    finally:
        driver.quit()


# ---------- 3) เรียก Gemini (Structured Output) ----------
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "symbol": {"type": "string"},
        "price": {"type": "number"},
        "change": {"type": "number"},
        "percent_change": {"type": "number"},
    },
    "required": ["symbol", "price", "change", "percent_change"],
    "additionalProperties": False,
}

EXTRACTION_PROMPT = """
คุณได้รับหน้าเว็บของตลาดหลักทรัพย์แห่งประเทศไทย (SET) เกี่ยวกับหุ้น MALEE
งานของคุณคือสกัดค่าต่อไปนี้จากข้อมูลหน้าเว็บ (ข้อความและภาพสกรีนช็อต):

- price: ราคาหุ้นปัจจุบันของ MALEE (ตัวเลขทศนิยม)
- change: การเปลี่ยนแปลงแบบจำนวน (เช่น +0.10 หรือ -0.05) ให้คืนเป็นตัวเลขทศนิยม
- percent_change: การเปลี่ยนแปลงแบบเปอร์เซ็นต์ (เช่น +0.70%) ให้คืนเป็นตัวเลขทศนิยม **ไม่ต้องใส่เครื่องหมาย %**
- symbol: ให้คืนเป็น "MALEE"

หลักการ:
1) ใช้บริบทที่แสดงเป็น "ราคาปัจจุบัน" มากที่สุด
2) ลบเครื่องหมาย % และวงเล็บ, รองรับ unicode minus
3) ตอบเป็น JSON ตามสคีมาที่กำหนดเท่านั้น
"""

def extract_fields_with_gemini(page_text: str, screenshot_path: str) -> dict:
    image_part = {
        "mime_type": "image/png",
        "data": open(screenshot_path, "rb").read()
    }

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    generation_config = {
        "temperature": 0,
        "response_mime_type": "application/json",
        "response_schema": RESPONSE_SCHEMA
    }

    resp = model.generate_content(
        contents=[
            EXTRACTION_PROMPT,
            {"mime_type": "text/plain", "text": page_text},
            image_part
        ],
        generation_config=generation_config,
        safety_settings=None,
    )

    data = json.loads(resp.text)

    # ปรับรูปแบบตัวเลขให้ชัวร์ (กันกรณีส่งมาเป็นสตริง)
    def to_float(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            s = x.strip().replace(",", "").replace("%", "").replace("(", "").replace(")", "").replace("−", "-")
            return float(s)
        raise ValueError(f"ไม่สามารถแปลงเป็น float: {x}")

    data["symbol"] = SYMBOL
    data["price"] = to_float(data["price"])
    data["change"] = to_float(data["change"])
    data["percent_change"] = to_float(data["percent_change"])
    return data


# ---------- 4) ประกอบผลลัพธ์สุดท้าย + timestamp ----------
def build_scraped_data(extracted: dict) -> dict:
    bkk_tz = pytz.timezone("Asia/Bangkok")
    ts = datetime.now(bkk_tz).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "symbol": SYMBOL,
        "price": extracted["price"],
        "change": extracted["change"],
        "percent_change": extracted["percent_change"],
        "timestamp": ts,
    }


# ---------- 5) บันทึกไฟล์ ----------
def save_to_file(data: dict, filename: str = "price.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filename}")


# ---------- 6) main ----------
if __name__ == "__main__":
    print(f"Rendering page: {URL}")
    configure_gemini()
    page_text, screenshot_file = render_set_page(URL)
    extracted = extract_fields_with_gemini(page_text, screenshot_file)
    scraped_data = build_scraped_data(extracted)
    print(json.dumps(scraped_data, ensure_ascii=False, indent=2))
    save_to_file(scraped_data, "price.json")
