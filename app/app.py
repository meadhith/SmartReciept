import re
from pathlib import Path

import cv2
import easyocr
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="SmartReceipt", page_icon="🧾")


@st.cache_resource
def load_models():
    yolo = YOLO(str(ROOT / "models" / "yolo_best.pt"))
    reader = easyocr.Reader(["en"], gpu=False)
    clf = joblib.load(ROOT / "models" / "classifier.joblib")["model"]
    return yolo, reader, clf


def amounts(text):
    return [float(x) for x in re.findall(r"\d+\.\d{2}", str(text).replace(",", ""))]


DATE_RE = r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{1,2}[ -]?[A-Za-z]{3}[ -]?\d{2,4}"


def run_pipeline(img, yolo, reader):
    h, w = img.shape[:2]
    res = yolo(cv2.resize(img, (640, 640)), conf=0.1, verbose=False)[0]

    fields, boxes = {}, []
    for b in res.boxes:
        name, conf = res.names[int(b.cls)], float(b.conf)
        if name in fields and fields[name]["conf"] >= conf:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        x1, x2 = int(x1 * w / 640), int(x2 * w / 640)
        y1, y2 = int(y1 * h / 640), int(y2 * h / 640)
        crop = img[max(y1, 0):y2, max(x1, 0):x2]
        text = " ".join(reader.readtext(crop, detail=0))
        fields[name] = {"conf": conf, "text": text, "box": (x1, y1, x2, y2)}

    lines = [l for l in reader.readtext(img, detail=0) if l.strip()]
    full_text = " ".join(lines)
    m = re.search(DATE_RE, full_text)
    amts = amounts(full_text)

    vendor = fields.get("vendor", {}).get("text") or (lines[0] if lines else "")
    date = (m.group(0) if m else "") or fields.get("date", {}).get("text", "")
    yolo_total = fields.get("total", {}).get("text", "")
    total = yolo_total if amounts(yolo_total) else (f"{max(amts):.2f}" if amts else "")
    return {"vendor": vendor, "date": date, "total": total}, fields


st.title("🧾 SmartReceipt")
st.write("Upload a receipt photo to extract the vendor, date and total, and predict the expense category.")

file = st.file_uploader("Receipt image", type=["jpg", "jpeg", "png"])

if file:
    pil = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    pil.thumbnail((1600, 1600))
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    with st.spinner("Loading models (first run takes a minute)..."):
        yolo, reader, clf = load_models()
    with st.spinner("Reading receipt..."):
        result, fields = run_pipeline(img, yolo, reader)

    shown = img.copy()
    for name, f in fields.items():
        x1, y1, x2, y2 = f["box"]
        cv2.rectangle(shown, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(shown, name, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected fields")
        st.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Extracted data")
        st.write(f"**Vendor:** {result['vendor'] or 'not found'}")
        st.write(f"**Date:** {result['date'] or 'not found'}")
        st.write(f"**Total:** {result['total'] or 'not found'}")
        if result["vendor"]:
            category = clf.predict([result["vendor"]])[0]
            st.success(f"Expense category: **{category}**")
        else:
            st.warning("No vendor found, so the category can't be predicted.")