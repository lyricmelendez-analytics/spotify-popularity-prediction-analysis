# 🎵 Spotify Popularity Analysis Dashboard

An interactive data science and machine learning dashboard built with **Streamlit** that analyzes factors influencing Spotify track popularity using a global music dataset (2009–2025).

This project explores how **song duration, release timing, explicit content, and artist popularity** impact streaming success, and includes a **Random Forest model** to predict track popularity.

---

## 🚀 Project Overview

This dashboard allows users to explore Spotify data interactively and understand what drives song popularity.

It combines:
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature engineering
- Machine learning prediction

---

## 📊 Key Features

### 🎧 Data Exploration
- Filter songs by album release year
- Filter explicit vs non-explicit tracks
- Dynamic metrics updated in real time

### 📈 Visual Analytics
- Song popularity vs duration analysis
- Popularity decay over time
- Explicit vs non-explicit comparison
- Feature distributions and trends

### 🤖 Machine Learning Model
- Random Forest Regressor trained on Spotify features
- Predicts track popularity
- Feature importance analysis
- Model evaluation using:
  - R² Score  
  - MAE (Mean Absolute Error)  
  - RMSE (Root Mean Squared Error)

---

## 🧠 Key Insights

- Artist popularity is one of the strongest predictors of streaming success
- Shorter songs tend to perform better due to replay behavior
- Newer releases significantly outperform older tracks
- Explicit content has minimal impact on popularity
- Release timing and album strategy are highly influential

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- Matplotlib
- Scikit-learn
- KaggleHub

---

## 📁 Project Structure
spotify-popularity-analysis/
├── popularity_streamlit_analysis.py # Streamlit dashboard
├── requirements.txt # Dependencies
├── README.md # Project documentation

---

## 📂 Dataset

**Spotify Global Music Dataset (2009–2025)**  
Source: Kaggle (via KaggleHub)

Includes:
- Track metadata
- Artist information
- Album data
- Popularity metrics
- Audio-related features

---


👤 Author

Lyric Melendez
Data Analyst | Python | Machine Learning | Media Analytics