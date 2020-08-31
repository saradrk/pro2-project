mkdir Data

mv Sarcasm_Headlines_Dataset_v2.json Data

python3 split_data.py

pip install -r requirements.txt

python -m spacy download en_core_web_sm

