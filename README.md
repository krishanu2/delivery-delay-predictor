# 🚚 Delivery Delay Predictor

A machine learning-powered dashboard to predict delays in delivery logistics and visualize key delivery performance insights.

## 🔍 Features

- Predicts whether a delivery will be delayed based on:
  - Region
  - Weather conditions
  - Distance
  - Dispatch delay
- Interactive dashboard with:
  - Delivery performance KPIs
  - Visuals like bar charts, scatter plots, and correlation heatmaps

## 📁 Files in This Repo

| File | Description |
|------|-------------|
| \pp.py\ | Streamlit web app for prediction + dashboard |
| \project.py\ | Data analysis and model training script |
| \model.pkl\ | Trained RandomForest model |
| \delivery_data.xlsx\ | Raw delivery dataset |
| \delivery_analysis_output.xlsx\ | Preprocessed delivery insights |
| \.png\ files | Generated plots (region, weather, heatmap, etc.) |

## ⚙️ How to Run

\\\ash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
\\\

## 📦 Requirements

- Python 3.9+
- Streamlit
- scikit-learn
- pandas
- seaborn
- matplotlib
- openpyxl
- haversine
- joblib

Generate requirements with:
\\\ash
pip freeze > requirements.txt
\\\

## 🙌 Author

**Krishanu Mahapatra**  
Made with ❤️ using Python and Streamlit.

[👉 Visit My GitHub Profile](https://github.com/krishanu2)

