# 🔍 Crime Forecasting: Spatiotemporal Prediction Using Machine Learning

This repository contains the complete implementation of our research paper:

📄 **"An Integrated Approach to Crime Prediction Using Time Series and Spatial Analysis"**  

---

## 📌 Overview

Urban crime is both **spatial** and **temporal** in nature. Our project introduces a **hybrid forecasting framework** that integrates:

- 📆 **Time Series Forecasting** using Prophet, STL, and LightGBM  
- 🗺️ **Spatial Prediction** using Random Forest with GIS coordinates  
- 🔁 **Stacked Ensemble Modeling** to improve forecast accuracy  
- 📊 **Visualization Dashboards**: Heatmaps, forecasts, EDA  

**Goal:** Predict daily crime counts and locate high-risk grid areas in Chicago using public crime data.

---

## 📂 Repository Structure

```
├── data/
│   └── chicago_crime_data.csv         # Cleaned dataset (2021–2025)
│
├── notebooks/
│   ├── ML_INTERNSHIP.ipynb           # Full training pipeline (time + spatial)
│   └── EDA_Heatmaps.ipynb            # Exploratory Data Analysis and Heatmaps
│
├── figures/
│   ├── stl_decomposition.png
│   ├── crime_heatmaps.png
│   ├── top10_crimes.png
│   └── forecast_comparison.png
│
├── models/
│   └── saved_models/                 # Serialized Prophet, LightGBM, RF models
│
├── utils/
│   └── preprocessing.py              # Feature engineering and cleaning
│
├── README.md                         # Project summary and setup guide
├── requirements.txt                  # Python dependencies
└── LICENSE
```

---

## 📈 Methodology

### 🔧 Preprocessing
- Handle missing values, drop duplicates
- Isolation Forest for outlier detection (1% contamination)
- DBSCAN for spatial clustering

### 🧠 Feature Engineering
- **Temporal:** lag features, rolling stats, STL decomposition
- **Spatial:** encode grid coordinates (H3 or manual grid)

### 📉 Modeling Techniques
- **📊 Prophet:** Long-term and seasonal crime trends
- **🌲 LightGBM:** Nonlinear pattern learning
- **🌍 Random Forest:** Grid-level spatial classification
- **🔁 Stacked Ensemble:** Combines base predictions (GBR as meta-learner)

### 📏 Evaluation Metrics
- R², MAE, MAPE, RMSE, Accuracy ±1 count

### 📊 Visualization
- Time series plots, STL components
- Spatial probability heatmaps for crime risk

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.8+
- Jupyter Notebook or Jupyter Lab

### ⚙️ Installation

```bash
# Clone this repository
git clone https://github.com/Shoaib-1302/Crime_Prediction.git
cd Crime_Prediction

# Install dependencies
pip install -r requirements.txt
```

### ▶️ Running the Pipeline

```bash
# Open Jupyter Lab or Notebook
jupyter lab
```

Then open and run:
```
notebooks/ML_INTERNSHIP.ipynb
```

---

## 📌 Key Results

| Model         | R² Score | MAE   | MAPE    |
|---------------|----------|-------|---------|
| Prophet       | 0.333    | 35.4  | ~       |
| LightGBM      | 0.322    | 35.44 | ~       |
| Random Forest | 0.356    | 0.16  | —       |
| **Ensemble**  | **0.966**| 5.86  | 1.36%   |

---

## 🌍 Visual Insights

- 🔥 **High-risk areas** of crime forecasted using our spatial model  
- 📈 **Comparison of actual vs predicted** crime count using Ensemble model

---

## 🤝 Contributors

- **Shoaib** – Time series forecasting, ensemble design  
- **Chittesh K** – Geospatial modeling, Random Forest classifier  
- **Deepa S** – Research guidance, model evaluation  
- **Rashmi Siddalingappa** – Review and editorial  
- **Vinay M** – Project supervision

---

## 📬 Contact

📧 shoaib@msds.christuniversity.in  
📧 chittesh.k@msds.christuniversity.in  

---

## 📜 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
