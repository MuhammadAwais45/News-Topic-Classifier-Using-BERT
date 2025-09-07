# 📰 News Topic Classifier (BERT + AG News)

A transformer-based text classification project that fine-tunes **BERT (`bert-base-uncased`)** on the **AG News dataset** to classify news headlines into four categories:

- 🌍 **World**  
- 🏟️ **Sports**  
- 💼 **Business**  
- 🔬 **Sci/Tech**  

Built with **Hugging Face Transformers**, **PyTorch**, and deployed using **Streamlit**.  

---

## ⚡ Features
✅ Fine-tunes **BERT** on the AG News dataset  
✅ Achieves high **Accuracy & F1-score** on the test set  
✅ **Interactive Streamlit GUI** for headline classification  
✅ Supports both **CPU & GPU (CUDA)** seamlessly  

---

## 📊 Dataset
- **AG News Dataset** (via Hugging Face Datasets)  
- ~120k training samples, ~7.6k test samples  
- 4 balanced categories  

---

## 🔮 Example Predictions
- `"UN holds emergency meeting over border tensions."` → 🌍 **World**  
- `"Apple unveils new chip to boost AI performance."` → 🔬 **Sci/Tech**  
- `"Manchester City clinch dramatic league title."` → 🏟️ **Sports**  
- `"Oil prices drop after surprising inventory data."` → 💼 **Business**  

---

## 📈 Evaluation Metrics
- **Accuracy**: ~94%  
- **F1-Score**: High across all 4 classes  

---

## 🎨 WordCloud Visualization
During EDA, we generated **WordClouds** per category to visualize frequent terms in the dataset.  

---

## ⚙️ Requirements
```bash
Python 3.8+
torch
transformers
datasets
streamlit
scikit-learn
matplotlib
wordcloud
```

## 📌 Skills Gained

🧠 NLP with Transformers
🔁 Fine-tuning & Transfer Learning
📊 Model Evaluation (Accuracy, F1)
🌐 Deployment with Streamlit
