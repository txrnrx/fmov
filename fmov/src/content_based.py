"""
Content-Based Filtering Module
==============================
TF-IDF vectorization and cosine similarity for content recommendations.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import os
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class ContentBasedRecommender:
    """Content-Based Filtering using TF-IDF and Cosine Similarity."""
    
    def __init__(self):
        self.shows_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.title_to_idx = {}
        self.is_fitted = False
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def fit(self, shows_df: pd.DataFrame, max_features: int = 5000) -> None:
        """Fit the content-based model using TF-IDF."""
        self.shows_df = shows_df.reset_index(drop=True)
        
        if 'combined_features' in shows_df.columns:
            content_features = shows_df['combined_features']
        else:
            cols = ['listed_in', 'description', 'director', 'cast']
            cols = [c for c in cols if c in shows_df.columns]
            content_features = shows_df[cols].fillna('').agg(' '.join, axis=1)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words='english',
            ngram_range=(1, 2), min_df=2, max_df=0.95
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_features)
        self.title_to_idx = {t.lower(): i for i, t in enumerate(shows_df['title'])}
        self.is_fitted = True
        print(f"✅ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def get_similar_by_title(self, title: str, n: int = 10) -> List[Dict]:
        """Get similar items by title."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        title_lower = title.lower()
        if title_lower not in self.title_to_idx:
            matches = [t for t in self.title_to_idx.keys() if title_lower in t]
            if matches:
                title_lower = matches[0]
            else:
                raise ValueError(f"Title '{title}' not found")
        
        idx = self.title_to_idx[title_lower]
        similarities = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        similar_indices = similarities.argsort()[::-1][1:n+1]
        
        results = []
        for sim_idx in similar_indices:
            show = self.shows_df.iloc[sim_idx]
            results.append({
                'content_id': show['content_id'], 'title': show['title'],
                'type': show['type'], 'genre': show['listed_in'],
                'release_year': show['release_year'],
                'similarity_score': round(float(similarities[sim_idx]), 4)
            })
        return results
    
    def get_similar_by_content_id(self, content_id: int, n: int = 10) -> List[Dict]:
        """Get similar items by content ID."""
        idx = self.shows_df[self.shows_df['content_id'] == content_id].index
        if len(idx) == 0:
            raise ValueError(f"Content ID {content_id} not found")
        idx = idx[0]
        similarities = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        similar_indices = similarities.argsort()[::-1][1:n+1]
        
        results = []
        for sim_idx in similar_indices:
            show = self.shows_df.iloc[sim_idx]
            results.append({
                'content_id': show['content_id'], 'title': show['title'],
                'type': show['type'], 'genre': show['listed_in'],
                'release_year': show['release_year'],
                'similarity_score': round(float(similarities[sim_idx]), 4)
            })
        return results
    
    def get_recommendations_for_user(self, user_id: int, ratings_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get content-based recommendations for a user."""
        user_ratings = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]
        if len(user_ratings) == 0:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        if len(user_ratings) == 0:
            return pd.DataFrame()
        
        all_similar = {}
        rated_ids = set(ratings_df[ratings_df['user_id'] == user_id]['content_id'])
        
        for _, row in user_ratings.iterrows():
            try:
                similar = self.get_similar_by_content_id(row['content_id'], n=20)
                for item in similar:
                    if item['content_id'] not in rated_ids:
                        cid = item['content_id']
                        score = item['similarity_score'] * (row['rating'] / 5.0)
                        if cid in all_similar:
                            all_similar[cid]['score'] += score
                            all_similar[cid]['count'] += 1
                        else:
                            all_similar[cid] = {**item, 'score': score, 'count': 1}
            except:
                continue
        
        for cid in all_similar:
            all_similar[cid]['final_score'] = all_similar[cid]['score'] / all_similar[cid]['count']
        
        recs = sorted(all_similar.values(), key=lambda x: x['final_score'], reverse=True)[:n]
        return pd.DataFrame(recs)
    
    def get_genre_recommendations(self, genres: List[str], n: int = 10) -> pd.DataFrame:
        """Get recommendations based on genres (cold-start solution)."""
        query = ' '.join(genres)
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:n]
        
        results = []
        for idx in top_indices:
            show = self.shows_df.iloc[idx]
            results.append({
                'content_id': show['content_id'], 'title': show['title'],
                'type': show['type'], 'genre': show['listed_in'],
                'release_year': show['release_year'],
                'relevance_score': round(float(similarities[idx]), 4)
            })
        return pd.DataFrame(results)
    
    def save_model(self, filename: str = "content_based_model.pkl") -> str:
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump({'vectorizer': self.tfidf_vectorizer, 'matrix': self.tfidf_matrix,
                     'title_to_idx': self.title_to_idx}, filepath)
        print(f"✅ Model saved to {filepath}")
        return filepath
    
    def load_model(self, filename: str = "content_based_model.pkl") -> None:
        filepath = os.path.join(self.model_dir, filename)
        data = joblib.load(filepath)
        self.tfidf_vectorizer = data['vectorizer']
        self.tfidf_matrix = data['matrix']
        self.title_to_idx = data['title_to_idx']
        self.is_fitted = True
