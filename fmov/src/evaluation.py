"""
Evaluation Module
=================
Metrics for evaluating recommendation system performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class RecommenderEvaluator:
    """Comprehensive evaluation metrics for recommendation systems."""
    
    def __init__(self, ratings_df: pd.DataFrame, shows_df: pd.DataFrame):
        self.ratings_df = ratings_df
        self.shows_df = shows_df
        self.threshold = 4  # Rating threshold for "relevant" items
    
    def rmse(self, predictions: List[Tuple]) -> float:
        """Calculate Root Mean Square Error."""
        if not predictions:
            return float('inf')
        
        squared_errors = [(true - pred) ** 2 for _, _, true, pred in predictions]
        return np.sqrt(np.mean(squared_errors))
    
    def mae(self, predictions: List[Tuple]) -> float:
        """Calculate Mean Absolute Error."""
        if not predictions:
            return float('inf')
        
        abs_errors = [abs(true - pred) for _, _, true, pred in predictions]
        return np.mean(abs_errors)
    
    def precision_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate Precision@K."""
        if k <= 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = sum(1 for item in recommended_k if item in relevant_set)
        
        return hits / k
    
    def recall_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_set = set(relevant)
        hits = sum(1 for item in recommended_k if item in relevant_set)
        
        return hits / len(relevant)
    
    def f1_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """Calculate F1@K."""
        precision = self.precision_at_k(recommended, relevant, k)
        recall = self.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def ndcg_at_k(self, recommended: List[int], relevant: Dict[int, float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if k <= 0 or not relevant:
            return 0.0
        
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                dcg += relevant[item] / np.log2(i + 2)
        
        ideal_gains = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mean_average_precision(self, user_recs: Dict[int, List[int]], 
                                user_relevant: Dict[int, List[int]], k: int) -> float:
        """Calculate Mean Average Precision (MAP)."""
        aps = []
        
        for user_id in user_recs:
            if user_id not in user_relevant:
                continue
            
            recommended = user_recs[user_id][:k]
            relevant = set(user_relevant[user_id])
            
            hits = 0
            precision_sum = 0
            
            for i, item in enumerate(recommended):
                if item in relevant:
                    hits += 1
                    precision_sum += hits / (i + 1)
            
            ap = precision_sum / len(relevant) if relevant else 0
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def coverage(self, all_recommendations: List[int], total_items: int) -> float:
        """Calculate catalog coverage."""
        unique_recommended = len(set(all_recommendations))
        return unique_recommended / total_items if total_items > 0 else 0.0
    
    def evaluate_model(self, model, test_users: List[int] = None, 
                      k_values: List[int] = [5, 10, 20]) -> Dict:
        """Comprehensive model evaluation."""
        if test_users is None:
            test_users = self.ratings_df['user_id'].unique()[:50]
        
        results = {f'precision@{k}': [] for k in k_values}
        results.update({f'recall@{k}': [] for k in k_values})
        results.update({f'f1@{k}': [] for k in k_values})
        all_recommendations = []
        
        for user_id in test_users:
            user_ratings = self.ratings_df[
                (self.ratings_df['user_id'] == user_id) & 
                (self.ratings_df['rating'] >= self.threshold)
            ]
            relevant = user_ratings['content_id'].tolist()
            
            if not relevant:
                continue
            
            try:
                recs = model.get_top_n_recommendations(user_id, n=max(k_values))
                recommended = [r[0] for r in recs]
                all_recommendations.extend(recommended)
                
                for k in k_values:
                    results[f'precision@{k}'].append(
                        self.precision_at_k(recommended, relevant, k))
                    results[f'recall@{k}'].append(
                        self.recall_at_k(recommended, relevant, k))
                    results[f'f1@{k}'].append(
                        self.f1_at_k(recommended, relevant, k))
            except:
                continue
        
        final_results = {}
        for metric, values in results.items():
            if values:
                final_results[metric] = np.mean(values)
        
        final_results['coverage'] = self.coverage(
            all_recommendations, len(self.shows_df))
        
        return final_results
    
    def print_evaluation_report(self, results: Dict) -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 50)
        print("       📊 EVALUATION REPORT")
        print("=" * 50)
        
        for metric, value in results.items():
            print(f"   {metric}: {value:.4f}")
        
        print("=" * 50)


def train_test_split_ratings(ratings_df: pd.DataFrame, 
                             test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test sets."""
    np.random.seed(42)
    
    test_indices = np.random.choice(
        ratings_df.index, 
        size=int(len(ratings_df) * test_size), 
        replace=False
    )
    
    test_df = ratings_df.loc[test_indices]
    train_df = ratings_df.drop(test_indices)
    
    print(f"✅ Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df
