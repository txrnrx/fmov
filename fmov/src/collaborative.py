"""
Collaborative Filtering Module
==============================
Implements Matrix Factorization (SVD) using pure NumPy/Scikit-learn.
No external dependencies on scikit-surprise required.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import joblib
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class CollaborativeFilteringRecommender:
    """
    Collaborative Filtering using SVD (Singular Value Decomposition).
    Pure NumPy/Scikit-learn implementation.
    """
    
    def __init__(self, n_factors: int = 100, n_epochs: int = 30):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.model = None
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.global_mean = 0
        self.user_biases = {}
        self.item_biases = {}
        self.ratings_df = None
        self.shows_df = None
        self.is_trained = False
        self.train_ratings = None
        self.test_ratings = None
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def prepare_data(self, ratings_df: pd.DataFrame, shows_df: pd.DataFrame, 
                     test_size: float = 0.2) -> None:
        """Prepare data for training."""
        self.ratings_df = ratings_df
        self.shows_df = shows_df
        
        # Create mappings
        users = ratings_df['user_id'].unique()
        items = ratings_df['content_id'].unique()
        
        self.user_mapping = {u: i for i, u in enumerate(users)}
        self.item_mapping = {i: j for j, i in enumerate(items)}
        self.reverse_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.reverse_item_mapping = {j: i for i, j in self.item_mapping.items()}
        
        # Split data
        self.train_ratings, self.test_ratings = train_test_split(
            ratings_df, test_size=test_size, random_state=42
        )
        
        print(f"✅ Data prepared:")
        print(f"   • Training ratings: {len(self.train_ratings):,}")
        print(f"   • Test ratings: {len(self.test_ratings):,}")
        print(f"   • Users: {len(users):,}")
        print(f"   • Items: {len(items):,}")
    
    def _create_matrix(self, ratings: pd.DataFrame) -> np.ndarray:
        """Create user-item rating matrix."""
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)
        
        matrix = np.zeros((n_users, n_items))
        
        for _, row in ratings.iterrows():
            u = self.user_mapping.get(row['user_id'])
            i = self.item_mapping.get(row['content_id'])
            if u is not None and i is not None:
                matrix[u, i] = row['rating']
        
        return matrix
    
    def train(self) -> Dict:
        """Train the SVD model."""
        if self.train_ratings is None:
            raise ValueError("Please prepare data first using prepare_data()")
        
        print(f"\n🚀 Training SVD model with {self.n_factors} factors...")
        
        # Create rating matrix
        rating_matrix = self._create_matrix(self.train_ratings)
        
        # Calculate biases
        self.global_mean = self.train_ratings['rating'].mean()
        
        user_means = self.train_ratings.groupby('user_id')['rating'].mean()
        item_means = self.train_ratings.groupby('content_id')['rating'].mean()
        
        self.user_biases = {u: user_means.get(u, self.global_mean) - self.global_mean 
                           for u in self.user_mapping.keys()}
        self.item_biases = {i: item_means.get(i, self.global_mean) - self.global_mean 
                           for i in self.item_mapping.keys()}
        
        # Center the matrix
        centered_matrix = rating_matrix.copy()
        for u_idx in range(centered_matrix.shape[0]):
            for i_idx in range(centered_matrix.shape[1]):
                if centered_matrix[u_idx, i_idx] > 0:
                    u = self.reverse_user_mapping[u_idx]
                    i = self.reverse_item_mapping[i_idx]
                    centered_matrix[u_idx, i_idx] -= (
                        self.global_mean + 
                        self.user_biases.get(u, 0) + 
                        self.item_biases.get(i, 0)
                    )
        
        # Apply SVD
        self.model = TruncatedSVD(n_components=min(self.n_factors, min(centered_matrix.shape) - 1), 
                                   random_state=42)
        self.user_factors = self.model.fit_transform(centered_matrix)
        self.item_factors = self.model.components_.T
        
        # Mark as trained before evaluation
        self.is_trained = True
        
        # Evaluate on test set
        rmse, mae = self._evaluate(self.test_ratings)
        
        metrics = {
            'model_type': 'svd',
            'rmse': rmse,
            'mae': mae,
            'n_factors': self.n_factors,
            'n_epochs': self.n_epochs
        }
        
        print(f"✅ Training complete!")
        print(f"   • RMSE: {rmse:.4f}")
        print(f"   • MAE: {mae:.4f}")
        
        return metrics
    
    def _evaluate(self, test_df: pd.DataFrame) -> Tuple[float, float]:
        """Evaluate model on test data."""
        errors = []
        abs_errors = []
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            content_id = row['content_id']
            true_rating = row['rating']
            
            # Only evaluate if user and item exist in training data
            if user_id in self.user_mapping and content_id in self.item_mapping:
                pred = self.predict_rating(user_id, content_id)
                errors.append((true_rating - pred) ** 2)
                abs_errors.append(abs(true_rating - pred))
        
        if len(errors) == 0:
            # Fallback: use baseline prediction for all test items
            for _, row in test_df.iterrows():
                pred = self.global_mean + self.user_biases.get(row['user_id'], 0)
                errors.append((row['rating'] - pred) ** 2)
                abs_errors.append(abs(row['rating'] - pred))
        
        rmse = np.sqrt(np.mean(errors)) if errors else 1.0
        mae = np.mean(abs_errors) if abs_errors else 1.0
        
        return rmse, mae
    
    def cross_validate(self, cv: int = 5) -> Dict:
        """Perform cross-validation."""
        if self.ratings_df is None:
            raise ValueError("Please prepare data first")
        
        print(f"\n📊 Running {cv}-fold cross-validation...")
        
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        rmse_scores = []
        mae_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.ratings_df)):
            train_data = self.ratings_df.iloc[train_idx]
            test_data = self.ratings_df.iloc[test_idx]
            
            # Temporary training
            self.train_ratings = train_data
            rating_matrix = self._create_matrix(train_data)
            
            self.global_mean = train_data['rating'].mean()
            user_means = train_data.groupby('user_id')['rating'].mean()
            item_means = train_data.groupby('content_id')['rating'].mean()
            
            self.user_biases = {u: user_means.get(u, self.global_mean) - self.global_mean 
                               for u in self.user_mapping.keys()}
            self.item_biases = {i: item_means.get(i, self.global_mean) - self.global_mean 
                               for i in self.item_mapping.keys()}
            
            centered_matrix = rating_matrix.copy()
            for u_idx in range(centered_matrix.shape[0]):
                for i_idx in range(centered_matrix.shape[1]):
                    if centered_matrix[u_idx, i_idx] > 0:
                        u = self.reverse_user_mapping[u_idx]
                        i = self.reverse_item_mapping[i_idx]
                        centered_matrix[u_idx, i_idx] -= (
                            self.global_mean + 
                            self.user_biases.get(u, 0) + 
                            self.item_biases.get(i, 0)
                        )
            
            self.model = TruncatedSVD(n_components=min(self.n_factors, min(centered_matrix.shape) - 1),
                                       random_state=42)
            self.user_factors = self.model.fit_transform(centered_matrix)
            self.item_factors = self.model.components_.T
            self.is_trained = True
            
            rmse, mae = self._evaluate(test_data)
            rmse_scores.append(rmse)
            mae_scores.append(mae)
        
        cv_results = {
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'cv_folds': cv
        }
        
        print(f"✅ Cross-validation complete!")
        print(f"   • RMSE: {cv_results['rmse_mean']:.4f} (±{cv_results['rmse_std']:.4f})")
        print(f"   • MAE: {cv_results['mae_mean']:.4f} (±{cv_results['mae_std']:.4f})")
        
        return cv_results
    
    def predict_rating(self, user_id: int, content_id: int) -> float:
        """Predict rating for a user-item pair."""
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
        
        u_idx = self.user_mapping.get(user_id)
        i_idx = self.item_mapping.get(content_id)
        
        # Base prediction
        pred = self.global_mean
        pred += self.user_biases.get(user_id, 0)
        pred += self.item_biases.get(content_id, 0)
        
        # Add latent factor contribution if user and item are known
        if u_idx is not None and i_idx is not None:
            pred += np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        
        # Clip to rating range
        return np.clip(pred, 1, 5)
    
    def get_top_n_recommendations(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Get top-N recommendations for a user."""
        if not self.is_trained:
            raise ValueError("Model not trained. Please call train() first.")
        
        # Get all content IDs
        all_content_ids = set(self.shows_df['content_id'].values)
        
        # Get content already rated by user
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        rated_content = set(user_ratings['content_id'].values)
        
        # Get unrated content
        unrated_content = all_content_ids - rated_content
        
        # Predict ratings for unrated content
        predictions = []
        for content_id in unrated_content:
            pred_rating = self.predict_rating(user_id, content_id)
            predictions.append((content_id, pred_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]
    
    def get_recommendations_with_details(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Get recommendations with full show details."""
        top_n = self.get_top_n_recommendations(user_id, n)
        
        recommendations = []
        for content_id, pred_rating in top_n:
            show_matches = self.shows_df[self.shows_df['content_id'] == content_id]
            if len(show_matches) > 0:
                show = show_matches.iloc[0]
                recommendations.append({
                    'content_id': content_id,
                    'title': show['title'],
                    'type': show['type'],
                    'genre': show['listed_in'],
                    'release_year': show['release_year'],
                    'rating': show['rating'],
                    'predicted_score': round(pred_rating, 2)
                })
        
        return pd.DataFrame(recommendations)
    
    def get_similar_users(self, user_id: int, n: int = 5) -> List[int]:
        """Find similar users based on latent factors."""
        if not self.is_trained:
            return []
        
        u_idx = self.user_mapping.get(user_id)
        if u_idx is None:
            return []
        
        user_vec = self.user_factors[u_idx]
        
        similarities = []
        for other_idx in range(len(self.user_factors)):
            if other_idx != u_idx:
                other_vec = self.user_factors[other_idx]
                sim = np.dot(user_vec, other_vec) / (
                    np.linalg.norm(user_vec) * np.linalg.norm(other_vec) + 1e-8
                )
                other_user = self.reverse_user_mapping[other_idx]
                similarities.append((other_user, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [u for u, _ in similarities[:n]]
    
    def save_model(self, filename: str = "collaborative_model.pkl") -> str:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        filepath = os.path.join(self.model_dir, filename)
        joblib.dump({
            'model': self.model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'global_mean': self.global_mean,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
            'n_factors': self.n_factors
        }, filepath)
        
        print(f"✅ Model saved to {filepath}")
        return filepath
    
    def load_model(self, filename: str = "collaborative_model.pkl") -> None:
        """Load a previously trained model."""
        filepath = os.path.join(self.model_dir, filename)
        data = joblib.load(filepath)
        
        self.model = data['model']
        self.user_factors = data['user_factors']
        self.item_factors = data['item_factors']
        self.user_mapping = data['user_mapping']
        self.item_mapping = data['item_mapping']
        self.reverse_user_mapping = data['reverse_user_mapping']
        self.reverse_item_mapping = data['reverse_item_mapping']
        self.global_mean = data['global_mean']
        self.user_biases = data['user_biases']
        self.item_biases = data['item_biases']
        self.n_factors = data['n_factors']
        self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")
