import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================
# Load model + tokenizer
# ======================
save_dir = "news_classifier_bert_base"
tok = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForSequenceClassification.from_pretrained(save_dir)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Your label mapping (adjust if needed)
id2label = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# ======================
# Streamlit App
# ======================
st.title("📰 News Topic Classifier (BERT)")

st.write("Enter a news headline and get its predicted category.")

# Input box
user_input = st.text_area("News headline:", "Apple unveils new chip to boost AI performance.")

if st.button("Classify"):
    if user_input.strip() != "":
        # Encode
        enc = tok([user_input], return_tensors="pt", truncation=True, padding=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}

        # Predict
        with torch.no_grad():
            logits = model(**enc).logits
        pred = torch.argmax(logits, dim=1).item()
        label = id2label[pred]

        st.success(f"**Predicted Category:** {label}")
    else:
        st.warning("Please enter some text.")
