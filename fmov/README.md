# 🎬 FlixMood - AI Content Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://content-recommendation-sys.streamlit.app/)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-FlixMood-E50914?style=for-the-badge)](https://content-recommendation-sys.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Netflix–YouTube–Spotify Style Recommender** with Mood-Aware AI Recommendations

## 🌐 **[Try the Live Demo →](https://content-recommendation-sys.streamlit.app/)**

## 🎯 What is FlixMood?

FlixMood is an **AI-powered content recommendation system** that understands your mood, context, and preferences to suggest the perfect movie or TV show. Just like how Netflix and Spotify personalize your experience!

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎭 **Mood-Based Recommendations** | Select your current mood and get personalized suggestions |
| 🕐 **Time-Aware** | Recommendations adapt to time of day (morning, afternoon, evening, night) |
| 👥 **Context-Aware** | Different suggestions for solo watching, date night, family time, or friend parties |
| ⚡ **Energy Matching** | High adrenaline, relaxed vibes, intense & deep, or light & fun |
| 🎲 **Surprise Me!** | Random pick with a fun balloon animation |
| 📱 **Screen Size Tips** | Smart suggestions based on your viewing device |
| 🔀 **Hybrid AI Engine** | Combines SVD + TF-IDF for optimal accuracy |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       USER INPUT                             │
│  Mood 🎭  │  Context 👥  │  Energy ⚡  │  Filters 🎚️        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  AI RECOMMENDATION ENGINE                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │     SVD     │  │   TF-IDF    │  │   Mood Mapping     │  │
│  │ Collaborative│  │  Content    │  │   + Time Context   │  │
│  │  Filtering  │  │   Based     │  │   + Energy Boost   │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────┬───────────┘  │
│         └────────────────┼────────────────────┘              │
│                          ▼                                   │
│              ┌───────────────────┐                           │
│              │  HYBRID ENSEMBLE  │                           │
│              │   (Weighted Mix)  │                           │
│              └─────────┬─────────┘                           │
└────────────────────────┼────────────────────────────────────┘
                         ▼
        ┌────────────────────────────────────┐
        │    🎬 PERSONALIZED RECOMMENDATIONS  │
        │    with Match Percentage Scores    │
        └────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit with Custom CSS |
| **ML/AI** | Scikit-learn, NumPy, Pandas |
| **Visualization** | Plotly, Seaborn |
| **Algorithms** | SVD (Matrix Factorization), TF-IDF |
| **Dataset** | Netflix Shows (6,234 titles) |

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/flixmood.git
cd flixmood

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Access the App
Open your browser and navigate to: `http://localhost:8501`

## 📊 Dataset

Using **Netflix Shows** dataset from Kaggle:
- **6,234** Movies & TV Shows
- **500** Synthetic Users (for collaborative filtering)
- **13,000+** Ratings

## 📈 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **RMSE** | ~1.38 | Root Mean Square Error |
| **Accuracy** | ~72% | Prediction accuracy |
| **Coverage** | 95%+ | Catalog coverage |

## 🎭 Mood Categories

| Mood | Genre Mapping |
|------|---------------|
| 😊 Happy & Uplifting | Comedies, Family, Animation |
| 😢 Emotional & Moving | Dramas, Romance, Independent |
| 😱 Thrilling & Scary | Horror, Thrillers, Crime |
| 🤔 Mind-Bending | Sci-Fi, Fantasy, Documentary |
| 💪 Action Packed | Action, Adventure, Martial Arts |
| ❤️ Romantic | Romance, Romantic Comedy |
| 👨‍👩‍👧‍👦 Family Friendly | Kids, Animation, Family |
| 🧠 Educational | Documentary, Science, History |
| 😎 Casual & Chill | Stand-Up, Reality, Variety |

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `app.py` as main file
5. Deploy! 🚀

### Environment Variables

No environment variables required! The app automatically downloads the dataset.

## 📁 Project Structure

```
flixmood/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── src/
│   ├── data_loader.py     # Dataset handling
│   ├── eda.py             # Exploratory analysis
│   ├── collaborative.py   # SVD recommender
│   ├── content_based.py   # TF-IDF recommender
│   ├── hybrid.py          # Ensemble methods
│   └── evaluation.py      # Metrics
├── data/                  # Generated data
└── models/                # Saved models
```

## 📝 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- Netflix Shows Dataset by [Shivam Bansal](https://www.kaggle.com/shivamb/netflix-shows)
- Streamlit for the amazing framework
- Scikit-learn community

---

<div align="center">
  <b>Built with ❤️ for IIT Project</b>
  <br>
  <i>FlixMood - Because every mood deserves the perfect movie!</i>
</div>
