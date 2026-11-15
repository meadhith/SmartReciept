# SmartReceipt

Receipt reader and expense classifier.
Pipeline: YOLOv8 (field detection) > EasyOCR (text) > ML classifiers (category).

## Setup
pip install -r requirements-dev.txt

## Run the app
streamlit run app/app.py