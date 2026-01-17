"""
Hybrid Recommender Module
=========================
Combines Collaborative Filtering and Content-Based approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import os
import warnings
warnings.filterwarnings('ignore')


class HybridRecommender:
    """Hybrid recommender combining CF and CB approaches."""
    
    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.4):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = None
        self.cb_model = None
        self.shows_df = None
        self.ratings_df = None
    
    def fit(self, cf_model, cb_model, shows_df: pd.DataFrame, ratings_df: pd.DataFrame):
        """Fit the hybrid model with both sub-models."""
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.shows_df = shows_df
        self.ratings_df = ratings_df
        print("✅ Hybrid recommender initialized")
    
    def get_recommendations(self, user_id: int, n: int = 10, 
                           strategy: str = 'weighted') -> pd.DataFrame:
        """
        Get hybrid recommendations.
        
        Strategies:
        - 'weighted': Weighted combination of CF and CB scores
        - 'switching': Use CF for users with history, CB for cold-start
        - 'cascade': Use CB to filter, CF to rank
        """
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        is_cold_start = len(user_ratings) < 5
        
        if strategy == 'switching':
            if is_cold_start:
                return self._get_cb_recommendations(user_id, n)
            return self._get_cf_recommendations(user_id, n)
        
        elif strategy == 'cascade':
            return self._cascade_recommendations(user_id, n)
        
        else:  # weighted
            return self._weighted_recommendations(user_id, n)
    
    def _get_cf_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Get collaborative filtering recommendations."""
        try:
            recs = self.cf_model.get_recommendations_with_details(user_id, n)
            recs['source'] = 'collaborative'
            return recs
        except:
            return pd.DataFrame()
    
    def _get_cb_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Get content-based recommendations."""
        try:
            recs = self.cb_model.get_recommendations_for_user(user_id, self.ratings_df, n)
            recs['source'] = 'content_based'
            return recs
        except:
            return pd.DataFrame()
    
    def _weighted_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Combine CF and CB with weighted scores."""
        cf_recs = self._get_cf_recommendations(user_id, n * 2)
        cb_recs = self._get_cb_recommendations(user_id, n * 2)
        
        combined = {}
        
        if not cf_recs.empty and 'predicted_score' in cf_recs.columns:
            for _, row in cf_recs.iterrows():
                cid = row['content_id']
                combined[cid] = {
                    'content_id': cid, 'title': row['title'],
                    'type': row['type'], 'genre': row['genre'],
                    'cf_score': row['predicted_score'] / 5.0,
                    'cb_score': 0
                }
        
        if not cb_recs.empty:
            score_col = 'final_score' if 'final_score' in cb_recs.columns else 'relevance_score'
            for _, row in cb_recs.iterrows():
                cid = row['content_id']
                if cid in combined:
                    combined[cid]['cb_score'] = row.get(score_col, 0)
                else:
                    combined[cid] = {
                        'content_id': cid, 'title': row['title'],
                        'type': row['type'], 'genre': row['genre'],
                        'cf_score': 0, 'cb_score': row.get(score_col, 0)
                    }
        
        for cid in combined:
            combined[cid]['hybrid_score'] = (
                self.cf_weight * combined[cid]['cf_score'] +
                self.cb_weight * combined[cid]['cb_score']
            )
        
        recs = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)[:n]
        result = pd.DataFrame(recs)
        result['source'] = 'hybrid'
        return result
    
    def _cascade_recommendations(self, user_id: int, n: int) -> pd.DataFrame:
        """Use CB to filter candidates, CF to rank."""
        cb_recs = self._get_cb_recommendations(user_id, n * 3)
        
        if cb_recs.empty:
            return self._get_cf_recommendations(user_id, n)
        
        candidates = cb_recs['content_id'].tolist()
        
        scored = []
        for cid in candidates:
            try:
                score = self.cf_model.predict_rating(user_id, cid)
                show = self.shows_df[self.shows_df['content_id'] == cid].iloc[0]
                scored.append({
                    'content_id': cid, 'title': show['title'],
                    'type': show['type'], 'genre': show['listed_in'],
                    'hybrid_score': score
                })
            except:
                continue
        
        result = pd.DataFrame(sorted(scored, key=lambda x: x['hybrid_score'], reverse=True)[:n])
        result['source'] = 'cascade'
        return result
    
    def explain_recommendation(self, user_id: int, content_id: int) -> Dict:
        """Explain why an item was recommended."""
        explanation = {'content_id': content_id, 'factors': []}
        
        try:
            cf_score = self.cf_model.predict_rating(user_id, content_id)
            explanation['cf_score'] = cf_score
            if cf_score >= 4.0:
                explanation['factors'].append("Users with similar tastes rated this highly")
        except:
            pass
        
        try:
            similar = self.cf_model.get_similar_users(user_id, n=3)
            if similar:
                explanation['similar_users'] = similar
        except:
            pass
        
        try:
            show = self.shows_df[self.shows_df['content_id'] == content_id].iloc[0]
            user_ratings = self.ratings_df[
                (self.ratings_df['user_id'] == user_id) & 
                (self.ratings_df['rating'] >= 4)
            ]
            if len(user_ratings) > 0:
                explanation['factors'].append(f"Similar to shows you've enjoyed")
        except:
            pass
        
        return explanation
