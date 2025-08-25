# scraper_ai_or.py
# ดึงข้อมูล "วัน - เวลา" และ "Diesel" จากหน้า OR (PTTOR) ด้วยแนวทาง AI (ไม่อิง CSS selector)
# ต้องมี GOOGLE_API_KEY อยู่ใน env (เช่น GitHub Actions: secrets -> GOOGLE_API_KEY)

import os
import json
from datetime import datetime
import pytz

import google.generativeai as genai

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


URL = "https://www.pttor.com/news/oil-price"
SCREENSHOT_PATH = "or_oil.png"
OUTPUT_JSON = "or_diesel_prices.json"


# ---------- 1) ตั้งค่า Gemini ----------
def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ไม่พบ GOOGLE_API_KEY ใน environment. "
            "กรณีใช้งานบน GitHub Actions ให้สร้าง secret ชื่อ GOOGLE_API_KEY และ export เข้าสู่ env ของ job"
        )
    genai.configure(api_key=api_key)


# ---------- 2) เรนเดอร์หน้าเว็บ (ดึงข้อความ + สกรีนช็อต) ----------
def render_page(url: str, screenshot_path: str = SCREENSHOT_PATH) -> tuple[str, str]:
    """
    ใช้ Selenium Manager (ไม่ระบุ service) ให้หา driver เองอัตโนมัติ
    อ่าน CHROME_PATH ถ้า workflow ติดตั้ง Chrome ให้แล้ว
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1600,1400")

    chrome_path = os.getenv("CHROME_PATH")  # ได้มาจาก browser-actions/setup-chrome
    if chrome_path:
        options.binary_location = chrome_path

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)
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


# ---------- 3) Prompt + Schema ให้โมเดลสกัดเฉพาะคอลัมน์ที่ต้องการ ----------
# หมายเหตุ: ห้ามใส่ additionalProperties ใน schema (ไลบรารีไม่รองรับ)
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # เก็บสตริงตามหน้าเว็บ (มักเป็นรูปแบบ dd-MM-พ.ศ. HH:MM)
                    "date_time": {"type": "string"},
                    # ราคาดีเซล เป็นตัวเลขทศนิยม (ไม่ใส่เครื่องหมายอื่น)
                    "diesel": {"type": "number"}
                },
                "required": ["date_time", "diesel"]
            }
        }
    },
    "required": ["rows"]
}

EXTRACTION_PROMPT = """
คุณได้รับหน้าเว็บ https://www.pttor.com/news/oil-price ของ OR (PTTOR) ที่แสดง "ราคาน้ำมัน"
งานของคุณคือ สกัด "เฉพาะ" ตาราง (table) ราคาน้ำมัน และดึงข้อมูล 2 คอลัมน์นี้ออกมาเป็นแถว ๆ:

- "วัน - เวลา" (หัวตารางภาษาไทย, อาจเขียนว่า วัน - เวลา) ให้คืนค่าเป็น string ตามที่เห็นในตาราง
- "ดีเซล" หรือ "Diesel" (หัวคอลัมน์อาจเป็นภาษาไทย/อังกฤษรวมกัน เช่น "ดีเซล Diesel") ให้คืนเป็นตัวเลขทศนิยม (ไม่รวมเครื่องหมายหรือข้อความประกอบ)

ข้อกำหนด:
1) พิจารณาเฉพาะพื้นที่ที่เป็น "ตารางราคา" ไม่ใช่ราคา ณ ปัจจุบันที่อยู่ส่วนอื่นของหน้า
2) ถ้าในตารางมีหลายชนิดดีเซล (เช่น ดีเซล B7, B20 ฯลฯ) ให้เลือกคอลัมน์ที่ชื่อว่า "ดีเซล" หรือ "Diesel" ที่เป็นคอลัมน์หลัก (ไม่ใช่อนุพันธ์)
3) คืนรูปแบบเป็น JSON ตรงตามสคีมาที่กำหนด โดย rows เป็นรายการของแถวตามลำดับที่พบ
4) ค่าราคาให้แปลงเป็นตัวเลข (number) เท่านั้น
"""


def _get_response_text(resp) -> str:
    """ดึงตัวข้อความจาก response เผื่อบางครั้ง resp.text ว่าง"""
    if getattr(resp, "text", None):
        return resp.text
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", []) or []
        texts = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                texts.append(t)
        if texts:
            return "\n".join(texts)
    return ""


def extract_table_with_gemini(page_text: str, screenshot_path: str) -> list[dict]:
    """
    ส่ง prompt + (ข้อความหน้าเว็บ + screenshot) เข้า Gemini
    เพื่อให้รีเทิร์น rows = [{date_time, diesel}, ...]
    """
    with open(screenshot_path, "rb") as f:
        image_bytes = f.read()

    image_part = {
        "inline_data": {
            "mime_type": "image/png",
            "data": image_bytes
        }
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
            page_text,
            image_part
        ],
        generation_config=generation_config,
        safety_settings=None,
    )

    raw = _get_response_text(resp)
    if not raw:
        raise RuntimeError("โมเดลไม่ส่งข้อความตอบกลับมา")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # กันเคสมี code fence
        raw2 = raw.strip()
        if raw2.startswith("```json"):
            raw2 = raw2[len("```json"):].strip()
        if raw2.startswith("```"):
            raw2 = raw2[len("```"):].strip()
        if raw2.endswith("```"):
            raw2 = raw2[:-3].strip()
        data = json.loads(raw2)

    rows = data.get("rows", [])
    # ทำความสะอาดผลลัพธ์ (กันกรณีโมเดลส่งสตริงของตัวเลขมา)
    cleaned = []
    for r in rows:
        dt = str(r.get("date_time", "")).strip()
        diesel = r.get("diesel", None)
        if isinstance(diesel, str):
            diesel = (
                diesel.strip()
                      .replace(",", "")
                      .replace("บาท", "")
                      .replace("฿", "")
            )
            try:
                diesel = float(diesel)
            except Exception:
                continue
        if dt and isinstance(diesel, (int, float)):
            cleaned.append({
                "date_time": dt,
                "diesel": float(diesel)
            })
    return cleaned


# ---------- 4) Build และบันทึกผล ----------
def build_output(rows: list[dict]) -> dict:
    bkk = pytz.timezone("Asia/Bangkok")
    now_bkk = datetime.now(bkk).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "source_url": URL,
        "scraped_at": now_bkk,
        "rows": rows
    }


def save_to_file(data: dict, filename: str = OUTPUT_JSON):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {filename}")


# ---------- 5) main ----------
if __name__ == "__main__":
    print(f"Rendering page: {URL}")
    configure_gemini()
    text, shot = render_page(URL)
    rows = extract_table_with_gemini(text, shot)
    out = build_output(rows)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_to_file(out, OUTPUT_JSON)
