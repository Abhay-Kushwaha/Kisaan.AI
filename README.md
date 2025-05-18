# Kisaan.AI

Kisaan.AI is an integrated AI-powered platform designed to empower Indian farmers and rural communities with smart agriculture and health solutions. It leverages machine learning and generative AI to provide Crop Recommendations, Crop Price Predictions, Fertilizer Suggestions, and Disease Prediction.


## 🚀 Features

- **🌾 Crop Price Predictor:**  
  Forecasts Prices for various crops using historical data, weather, expences and regional trends.

- **🌿 Crop Recommendation System:**  
  Suggests the most suitable crop to cultivate based on soil nutrients, pH, climate, and environmental factors.

- **🌱 Fertilizer Recommendation:**  
  Recommends optimal fertilizers by analyzing soil NPK, temperature, moisture, and crop requirements.

- **🩺 Multi-Disease Prediction:**  
  Predicts the likelihood of common health issues (Allergies, Hypertension, Gastric Problems, Migraine, and more) using user symptoms and ML models. Gives personalized health advices, diet suggestions, precautions and medication recommendations.

- **📊 Analytics Dashboard:**  
  Visualizes model performance, confusion matrices, residuals, and more for transparency and trust.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML5, Bootstrap 5, JavaScript
- **ML/AI:** scikit-learn, Google Gemini (Generative AI), gTTS (Text-to-Speech)
- **Data:** Pandas, CSV datasets
- **Visualization:** Matplotlib
- **Deployment:** Localhost (can be deployed to cloud platforms)

## 📂 Project Structure

```
Final Project/
│
├── app.py                  # Main Flask application
├── disease.py              # Disease prediction logic
├── mapping.py              # Analytics data mapping
├── model/                  # Trained ML models (pickle files)
├── dataset/                # Datasets for training
├── analytics/              # Model metrics (JSON)
├── maps/                   # Model plots (PNG)
├── static/
│   ├── audio/              # Generated audio files
│   └── img/                # Images and videos
├── templates/
│   ├── index.html
│   ├── crop.html
│   ├── crop_price.html
│   ├── fertilizer.html
│   ├── disease.html
│   ├── analytics.html
│   ├── faq.html
│   └── about.html
├── kisaan_env/
└── README.md
```

## ⚡ Quick Start

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Abhay-Kushwaha/Kisaan.AI
    cd Final Project
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up your Gemini API key:**
    - Create a file at `kisaan_env/key.txt` with the content:
      ```
      API_KEY=your_actual_google_api_key
      ```

4. **Run the application:**
    ```sh
    python app.py
    ```
    - Open [http://localhost:5000](http://localhost:5000) in your browser.


## 📝 Usage

- **Home:** Explore all modules from the dashboard.
- **Crop Recommendation:** Enter soil and climate details to get the best crop suggestion and a detailed explanation.
- **Crop Price Prediction:** Select crop and region details to estimate market price and profit margin.
- **Fertilizer Recommendation:** Input soil and crop info to get the best fertilizer advice.
- **Disease Prediction:** Select symptoms to get possible health issues, precautions, and AI-generated advice (with speech).
- **Analytics:** View model performance, confusion matrices, and more.

## 🤝 Collaborators

- **Name:** Abhay Kushwaha (2202900100006)
- **Role:** Backend, AI/ML Engineer
- **LinkedIn:** https://www.linkedin.com/in/abhay-k-5a0902278/
- **GitHub:** https://github.com/Abhay-Kushwaha
---
- **Name:** Aakash Jha (2202900100001)
- **Role:** Backend, AI/ML Engineer
- **LinkedIn:** https://www.linkedin.com/in/aakash-jha-a11931257/
- **GitHub:** https://github.com/Aakash-Jha3903
---
- **Name:** Akanshu Mittal (2202900100019)
- **Role:** Frontend, AI/ML Engineer
- **LinkedIn:** https://www.linkedin.com/in/akanshu-mittal-527130251/
- **GitHub:** https://github.com/AkanshuMittal

## 📢 Acknowledgements

- Datasets from [Kaggle](https://www.kaggle.com/)
- Google Gemini API for generative AI.
- scikit-learn, Flask, Bootstrap, and the open-source community.

## 📜 License

This project is for educational and research purposes only.