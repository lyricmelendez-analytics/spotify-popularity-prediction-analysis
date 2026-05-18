import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Spotify Popularity Analysis",
    layout="wide"
)

st.title("Spotify Song Popularity Analysis")
st.markdown(
    """
This dashboard analyzes the factors that influence Spotify song popularity,
with a focus on song duration, release timing, artist popularity,
and album strategy.
"""
)

# =====================================
# LOAD DATA
# =====================================

@st.cache_data

def load_data():

    path = kagglehub.dataset_download(
        "wardabilal/spotify-global-music-dataset-20092025"
    )

    df = pd.read_csv(f"{path}/spotify_data clean.csv")

    return df

spotify_data = load_data()

st.success("Dataset loaded successfully.")

# =====================================
# DATA CLEANING
# =====================================

spotify_data['album_release_date'] = pd.to_datetime(
    spotify_data['album_release_date'],
    errors='coerce'
)

spotify_data['album_release_year'] = (
    spotify_data['album_release_date'].dt.year
)

spotify_data['days_since_album_release'] = (
    pd.Timestamp(datetime.now()) - spotify_data['album_release_date']
).dt.days

spotify_data['explicit'] = spotify_data['explicit'].astype(int)

# Genre count feature
spotify_data['artist_genres'] = spotify_data['artist_genres'].fillna('')
spotify_data['genre_count'] = spotify_data['artist_genres'].apply(
    lambda x: len(str(x).split(',')) if x != '' else 0
)

# One-hot encode album type
album_type_dummies = pd.get_dummies(
    spotify_data['album_type'],
    prefix='album_type'
).astype(int)

spotify_data = pd.concat(
    [spotify_data, album_type_dummies],
    axis=1
)

# =====================================
# SIDEBAR FILTERS
# =====================================

st.sidebar.header("Filters")

min_year = int(spotify_data['album_release_year'].min())
max_year = int(spotify_data['album_release_year'].max())

selected_years = st.sidebar.slider(
    "Album Release Year",
    min_year,
    max_year,
    (2015, max_year)
)

explicit_filter = st.sidebar.selectbox(
    "Explicit Tracks",
    ["All", "Explicit Only", "Non-Explicit Only"]
)

filtered_data = spotify_data[
    (spotify_data['album_release_year'] >= selected_years[0]) &
    (spotify_data['album_release_year'] <= selected_years[1])
]

if explicit_filter == "Explicit Only":
    filtered_data = filtered_data[
        filtered_data['explicit'] == 1
    ]
elif explicit_filter == "Non-Explicit Only":
    filtered_data = filtered_data[
        filtered_data['explicit'] == 0
    ]

# =====================================
# OVERVIEW METRICS
# =====================================

st.header("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Tracks",
    f"{len(filtered_data):,}"
)

col2.metric(
    "Average Popularity",
    round(filtered_data['track_popularity'].mean(), 2)
)

col3.metric(
    "Average Duration (min)",
    round(filtered_data['track_duration_min'].mean(), 2)
)

col4.metric(
    "Average Artist Popularity",
    round(filtered_data['artist_popularity'].mean(), 2)
)

# =====================================
# SONG DURATION ANALYSIS
# =====================================

st.header("Song Popularity vs Duration")

st.markdown(
    """
This analysis investigates whether song length impacts Spotify popularity.
Streaming platforms may reward shorter songs due to replayability,
listener retention, and algorithmic behavior.
"""
)

# Duration buckets
filtered_data['duration_bucket'] = pd.cut(
    filtered_data['track_duration_min'],
    bins=[0, 2, 3, 4, 5, 10],
    labels=[
        '<2 min',
        '2-3 min',
        '3-4 min',
        '4-5 min',
        '5+ min'
    ]
)

bucket_popularity = filtered_data.groupby(
    'duration_bucket'
)['track_popularity'].mean()

fig1, ax1 = plt.subplots(figsize=(8, 5))

bucket_popularity.plot(kind='bar', ax=ax1)

ax1.set_xlabel('Track Duration')
ax1.set_ylabel('Average Popularity')
ax1.set_title('Average Popularity by Song Duration')

st.pyplot(fig1)

# Scatterplot
fig2, ax2 = plt.subplots(figsize=(10, 6))

sample_data = filtered_data.sample(
    min(5000, len(filtered_data)),
    random_state=42
)

ax2.scatter(
    sample_data['track_duration_min'],
    sample_data['track_popularity'],
    alpha=0.3
)

ax2.set_xlabel('Track Duration (Minutes)')
ax2.set_ylabel('Track Popularity')
ax2.set_title('Track Popularity vs Duration')

st.pyplot(fig2)

# =====================================
# RELEASE RECENCY ANALYSIS
# =====================================

st.header("Popularity Decay Over Time")

st.markdown(
    """
This section analyzes how popularity changes as albums age.
Streaming success is often heavily tied to recency.
"""
)

release_bins = [0, 30, 90, 180, 365, 730, 2000, 10000]
release_labels = [
    '<1 Month',
    '1-3 Months',
    '3-6 Months',
    '6-12 Months',
    '1-2 Years',
    '2-5 Years',
    '5+ Years'
]

filtered_data['release_age_bucket'] = pd.cut(
    filtered_data['days_since_album_release'],
    bins=release_bins,
    labels=release_labels
)

release_popularity = filtered_data.groupby(
    'release_age_bucket'
)['track_popularity'].mean()

fig3, ax3 = plt.subplots(figsize=(10, 5))

release_popularity.plot(kind='line', marker='o', ax=ax3)

ax3.set_xlabel('Album Age')
ax3.set_ylabel('Average Popularity')
ax3.set_title('Popularity Decay by Album Age')

st.pyplot(fig3)

# =====================================
# EXPLICIT ANALYSIS
# =====================================

st.header("Explicit vs Non-Explicit Tracks")

explicit_avg = filtered_data.groupby(
    'explicit'
)['track_popularity'].mean()

explicit_avg.index = ['Non-Explicit', 'Explicit']

fig4, ax4 = plt.subplots(figsize=(6, 5))

explicit_avg.plot(kind='bar', ax=ax4)

ax4.set_ylabel('Average Popularity')
ax4.set_title('Popularity: Explicit vs Non-Explicit')

st.pyplot(fig4)

# =====================================
# MACHINE LEARNING MODEL
# =====================================

st.header("Machine Learning Model")

st.markdown(
    """
A Random Forest Regressor was trained to predict Spotify track popularity.
The model identifies which features contribute most strongly to popularity.
"""
)

# Feature selection
exclude_columns = [
    'track_popularity',
    'track_id',
    'track_name',
    'artist_name',
    'artist_genres',
    'album_id',
    'album_name',
    'album_release_date',
    'album_type',
    'duration_bucket',
    'release_age_bucket'
]

exclude_columns = [
    col for col in exclude_columns
    if col in filtered_data.columns
]

features = filtered_data.drop(columns=exclude_columns)

features = features.select_dtypes(include=['number', 'bool'])

bool_cols = features.select_dtypes(include=['bool']).columns
features[bool_cols] = features[bool_cols].astype(int)

features = features.fillna(0)

target = filtered_data['track_popularity']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Metrics display
metric1, metric2, metric3 = st.columns(3)

metric1.metric("R² Score", round(r2, 3))
metric2.metric("MAE", round(mae, 3))
metric3.metric("RMSE", round(rmse, 3))

# Feature importance
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': features.columns,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(
    by='Importance',
    ascending=False
)

st.subheader("Feature Importance")

fig5, ax5 = plt.subplots(figsize=(10, 7))

top_features = feature_importance_df.head(15)

ax5.barh(
    top_features['Feature'],
    top_features['Importance']
)

ax5.set_xlabel('Importance')
ax5.set_ylabel('Feature')
ax5.set_title('Top Predictors of Spotify Track Popularity')

ax5.invert_yaxis()

st.pyplot(fig5)

# =====================================
# KEY INSIGHTS
# =====================================

st.header("Key Insights")

st.markdown(
    f"""
### Main Findings

- Artist popularity is one of the strongest predictors of streaming success.
- Song duration appears to meaningfully influence track popularity.
- More recent releases generally perform better on Spotify.
- Explicitness contributes less to popularity than expected.
- Album strategy and release timing impact streaming performance.

### Strategic Implications

- Emerging artists may benefit from shorter, replay-friendly songs.
- Consistent release cadence may improve streaming visibility.
- Artist ecosystem strength appears more important than individual track characteristics alone.

### Model Performance

- R² Score: {round(r2, 3)}
- MAE: {round(mae, 3)}
- RMSE: {round(rmse, 3)}
"""
)

# =====================================
# RAW DATA
# =====================================

st.header("Raw Dataset")

st.dataframe(filtered_data.head(100))