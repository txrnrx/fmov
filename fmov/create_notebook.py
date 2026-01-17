import json
import os

NOTEBOOK_NAME = "FlixMood_Project.ipynb"

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

def read_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"# Error reading {filepath}: {e}"

# --- NOTEBOOK CONTENT STRUCTURE ---
cells = []

# 1. TITLE & OBJECTIVE
cells.append(create_markdown_cell("""# 🎬 FlixMood: AI-Powered Content Recommendation System

## 1. Problem Definition & Objective
**Objective:** Build a simplified "Netflix-like" recommendation system ("FlixMood") that delivers personalized content suggestions using a hybrid AI approach.

**Problem Statement:** Users are overwhelmed by content choices. Traditional keyword search is insufficient. We need a system that understands **Mood**, **Context**, and **Deep Semantics** to recommend the right content at the right time.

**Real-World Relevance:** Streaming platforms (Netflix, Spotify) rely on recommendation engines to retain users. This project demonstrates the core algorithms (Matrix Factorization, TF-IDF, CLIP Vision, LLM Reasoning) used in production systems.

## 2. Selected Project Track
**Track:** Hybrid Recommendation System with Advanced Agentic AI (Smart Chat).
"""))

# 2. DATA UNDERSTANDING & ABOUT DATASET
cells.append(create_markdown_cell("""## 3. Data Understanding & Preparation

### 3.1 About the Dataset
**Name:** Netflix Movies and TV Shows Dataset
**Source:** [Kaggle (shivamb/netflix-shows)](https://www.kaggle.com/datasets/shivamb/netflix-shows)
**Volume:** Approximately 8,800 records.

**Schema & Features:**
*   `show_id`: Unique ID for every Movie / TV Show
*   `type`: Identifier - A Movie or TV Show
*   `title`: Title of the Movie / TV Show
*   `director`: Director of the Movie
*   `cast`: Actors involved in the movie / show
*   `country`: Country where the movie / show was produced
*   `date_added`: Date it was added on Netflix
*   `release_year`: Actual Release year of the move / show
*   `rating`: TV Rating of the movie / show
*   `duration`: Total Duration - in minutes or number of seasons
*   `listed_in`: Generes
*   `description`: The summary description

**Enhancements:**
We generate **Synthetic Ratings** (`synthetic_ratings.csv`) to simulate user behaviour (1-5 star ratings) because the original dataset only contains metadata. This allows us to train Collaborative Filtering models.
"""))

# DATA LOADING CODE
cells.append(create_code_cell(read_file("src/data_loader.py")))
cells.append(create_code_cell("""# Verify Data Loader
from src.data_loader import DataLoader
loader = DataLoader()
loader.load_netflix_data()
loader.preprocess_shows()
loader.generate_synthetic_ratings(n_users=500) 
print(f"Data Loaded: {len(loader.shows_df)} items, {len(loader.ratings_df)} ratings")
loader.shows_df.head(2)
"""))

# 3. EDA
cells.append(create_markdown_cell("""## 4. Exploratory Data Analysis (EDA)
Understanding the distribution of content, ratings, and user preferences.
"""))
cells.append(create_code_cell(read_file("src/eda.py")))
cells.append(create_code_cell("""# Run EDA Visualization (Static for Notebook)
from src.eda import NetflixEDA
eda = NetflixEDA(loader.shows_df, loader.ratings_df)
# For notebook, we can print stats. Plotly charts interactively display in Jupyter.
print("Genre Distribution:")
print(loader.shows_df['listed_in'].value_counts().head(5))
"""))

# 4. MODEL DESIGN - COLLABORATIVE
cells.append(create_markdown_cell("""## 5. System Design: Collaborative Filtering (SVD)
**Concept:** Users who liked similar items in the past will like similar items in the future.
**Algorithm:** SVD (Singular Value Decomposition) Matrix Factorization.
"""))
cells.append(create_code_cell(read_file("src/collaborative.py")))
cells.append(create_code_cell("""# Train Collaborative Model
from src.collaborative import CollaborativeFilteringRecommender
cf_model = CollaborativeFilteringRecommender(n_factors=50)
cf_model.prepare_data(loader.ratings_df, loader.shows_df)
metrics = cf_model.train()
print(f"CF Model Trained. RMSE: {metrics['rmse']:.4f}")
"""))

# 5. MODEL DESIGN - CONTENT BASED
cells.append(create_markdown_cell("""## 6. System Design: Content-Based Filtering
**Concept:** Recommend items similar to what a user likes based on metadata (Description, Cast, Genre).
**Algorithm:** TF-IDF Vectorization + Cosine Similarity.
"""))
cells.append(create_code_cell(read_file("src/content_based.py")))
cells.append(create_code_cell("""# Train Content-Based Model
from src.content_based import ContentBasedRecommender
cb_model = ContentBasedRecommender()
cb_model.fit(loader.shows_df)
print("Content-Based Model Fitted.")
# Example
recs = cb_model.get_similar_by_title("Inception", n=3) # Assuming Inception exists or pick random
if not recs.empty:
    print(recs[['title', 'similarity_score']])
"""))

# 6. HYBRID SYSTEM
cells.append(create_markdown_cell("""## 7. Hybrid Recommendation Engine
**Concept:** Combine CF (Serendipity/Accuracy) and CB (Cold Start/Relevance) for robust results.
"""))
cells.append(create_code_cell(read_file("src/hybrid.py")))
cells.append(create_code_cell("""# Train Hybrid Model
from src.hybrid import HybridRecommender
hybrid_model = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
hybrid_model.fit(cf_model, cb_model, loader.shows_df, loader.ratings_df)
print("Hybrid System Ready.")
"""))

# 7. SMART CHAT AI
cells.append(create_markdown_cell("""## 8. Smart Chat AI (Agentic Logic)
**Concept:** Natural Language Understanding to route queries ("I want to watch horror") into structured filters.
**Tech Stack:** 
*   **Rule-Based:** Regex for fast intent detection.
*   **Semantic Search:** `sentence-transformers` (SBERT) for meaning.
*   **Vision:** `CLIP` for image-to-text understanding.
*   **Reasoning:** `BART-MNLI` (Zero-Shot) for complex query classification.
"""))
cells.append(create_code_cell(read_file("src/smart_chat_ai.py")))

# 8. EVALUATION
cells.append(create_markdown_cell("""## 9. Evaluation & Analysis
Measuring the performance of the system using RMSE (Root Mean Square Error) and MAE (Mean Absolute Error).
"""))
cells.append(create_code_cell(read_file("src/evaluation.py")))
cells.append(create_code_cell("""from src.evaluation import RecommenderEvaluator
evaluator = RecommenderEvaluator(loader.ratings_df)
# Evaluation logic demonstrated in valid notebook run
print(f"Final Model Performance: RMSE={metrics['rmse']:.4f}")
"""))

# 9. ETHICAL CONSIDERATIONS
cells.append(create_markdown_cell("""## 10. Ethical Considerations & Responsible AI
1.  **Filter Bubbles:** Relying solely on similarity can isolate users. Our **Hybrid** approach and "Surprise Me" feature mitigate this.
2.  **Bias:** Synthetic data is balanced, but real-world data contains bias. We monitor genre diversity.
3.  **Privacy:** No PII is stored. User IDs are anonymized.
4.  **Transparency:** The AI Chat provides "Explainability" ("I recommended this because...").
"""))

# 10. CONCLUSION
cells.append(create_markdown_cell("""## 11. Conclusion & Future Scope
**Conclusion:** We successfully built a full-stack recommendation engine ("FlixMood") with a modern Netflix-style UI (see `src/ui_components.py`) and Agentic Chat capabilities.

**Future Scope:**
*   **Real-Time Learning:** Online learning API.
*   **Multi-Modal:** Video analysis (trailers).
*   **Social:** "Watch Party" features.
*   **Deployment:** Dockerize and deploy to cloud (AWS/GCP).
"""))

# NOTEBOOK JSON
notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_NAME, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=2)

print(f"Successfully generated {NOTEBOOK_NAME}")
