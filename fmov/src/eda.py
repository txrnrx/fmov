"""
Exploratory Data Analysis (EDA) Module
======================================
Comprehensive analysis and visualization of the Netflix dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import os
import warnings

warnings.filterwarnings('ignore')


class NetflixEDA:
    """
    Exploratory Data Analysis for Netflix Shows dataset.
    """
    
    def __init__(self, shows_df: pd.DataFrame, ratings_df: pd.DataFrame = None):
        self.shows_df = shows_df
        self.ratings_df = ratings_df
        self.figures_dir = "figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def content_type_analysis(self) -> go.Figure:
        """
        Analyze the distribution of Movies vs TV Shows.
        """
        type_counts = self.shows_df['type'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.4,
                marker_colors=['#E50914', '#564d4d'],
                textinfo='label+percent',
                textfont_size=14
            )
        ])
        
        fig.update_layout(
            title={
                'text': '🎬 Content Type Distribution',
                'font': {'size': 20}
            },
            template='plotly_dark'
        )
        
        return fig
    
    def release_year_trend(self) -> go.Figure:
        """
        Analyze content release trends over years.
        """
        year_counts = self.shows_df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
        year_counts = year_counts[year_counts.index >= 2000]  # Focus on recent years
        
        fig = go.Figure()
        
        colors = {'Movie': '#E50914', 'TV Show': '#00D9FF'}
        
        for content_type in year_counts.columns:
            fig.add_trace(go.Scatter(
                x=year_counts.index,
                y=year_counts[content_type],
                mode='lines+markers',
                name=content_type,
                line=dict(color=colors.get(content_type, '#FFFFFF'), width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title={'text': '📈 Content Release Trend by Year', 'font': {'size': 20}},
            xaxis_title='Release Year',
            yaxis_title='Number of Titles',
            template='plotly_dark',
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def genre_analysis(self) -> go.Figure:
        """
        Analyze genre distribution.
        """
        # Extract and count genres
        all_genres = []
        for genres in self.shows_df['listed_in'].dropna():
            all_genres.extend([g.strip() for g in genres.split(',')])
        
        genre_counts = Counter(all_genres)
        top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15])
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(top_genres.values()),
                y=list(top_genres.keys()),
                orientation='h',
                marker_color='#E50914',
                text=list(top_genres.values()),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title={'text': '🎭 Top 15 Genres', 'font': {'size': 20}},
            xaxis_title='Number of Titles',
            yaxis_title='Genre',
            template='plotly_dark',
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=20, r=20, t=60, b=40)
        )
        
        return fig
    
    def country_analysis(self) -> go.Figure:
        """
        Analyze content by country.
        """
        # Extract primary country
        self.shows_df['primary_country'] = self.shows_df['country'].apply(
            lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
        )
        
        country_counts = self.shows_df['primary_country'].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=country_counts.index,
                y=country_counts.values,
                marker_color=px.colors.sequential.Reds[::-1][:10],
                text=country_counts.values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title={'text': '🌍 Top 10 Countries by Content', 'font': {'size': 20}},
            xaxis_title='Country',
            yaxis_title='Number of Titles',
            template='plotly_dark',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def ratings_distribution(self) -> go.Figure:
        """
        Analyze content ratings distribution (age ratings).
        """
        rating_counts = self.shows_df['rating'].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=rating_counts.index,
                y=rating_counts.values,
                marker_color='#00D9FF',
                text=rating_counts.values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title={'text': '📺 Content Rating Distribution', 'font': {'size': 20}},
            xaxis_title='Rating',
            yaxis_title='Number of Titles',
            template='plotly_dark'
        )
        
        return fig
    
    def user_ratings_analysis(self) -> go.Figure:
        """
        Analyze synthetic user ratings distribution.
        """
        if self.ratings_df is None:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Rating Distribution', 'Ratings per User')
        )
        
        # Rating distribution
        rating_counts = self.ratings_df['rating'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=rating_counts.index,
                y=rating_counts.values,
                marker_color='#E50914',
                name='Rating Count'
            ),
            row=1, col=1
        )
        
        # Ratings per user
        user_rating_counts = self.ratings_df.groupby('user_id').size()
        fig.add_trace(
            go.Histogram(
                x=user_rating_counts,
                marker_color='#00D9FF',
                name='Users'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title={'text': '⭐ User Rating Analysis', 'font': {'size': 20}},
            template='plotly_dark',
            showlegend=False
        )
        
        return fig
    
    def sparsity_analysis(self) -> dict:
        """
        Analyze the sparsity of the user-item matrix.
        """
        if self.ratings_df is None:
            return None
        
        n_users = self.ratings_df['user_id'].nunique()
        n_items = self.ratings_df['content_id'].nunique()
        n_ratings = len(self.ratings_df)
        
        total_possible = n_users * n_items
        sparsity = 1 - (n_ratings / total_possible)
        
        return {
            'n_users': n_users,
            'n_items': n_items,
            'n_ratings': n_ratings,
            'total_possible': total_possible,
            'sparsity': sparsity,
            'sparsity_percent': f"{sparsity * 100:.2f}%"
        }
    
    def duration_analysis(self) -> go.Figure:
        """
        Analyze movie durations and TV show seasons.
        """
        movies = self.shows_df[self.shows_df['type'] == 'Movie'].copy()
        movies['duration_mins'] = movies['duration'].str.extract('(\d+)').astype(float)
        
        fig = go.Figure(data=[
            go.Histogram(
                x=movies['duration_mins'].dropna(),
                nbinsx=30,
                marker_color='#E50914'
            )
        ])
        
        fig.update_layout(
            title={'text': '⏱️ Movie Duration Distribution', 'font': {'size': 20}},
            xaxis_title='Duration (minutes)',
            yaxis_title='Count',
            template='plotly_dark'
        )
        
        return fig
    
    def generate_full_report(self) -> dict:
        """
        Generate all EDA visualizations and return them.
        """
        report = {
            'content_type': self.content_type_analysis(),
            'release_trend': self.release_year_trend(),
            'genres': self.genre_analysis(),
            'countries': self.country_analysis(),
            'content_ratings': self.ratings_distribution(),
            'duration': self.duration_analysis()
        }
        
        if self.ratings_df is not None:
            report['user_ratings'] = self.user_ratings_analysis()
            report['sparsity'] = self.sparsity_analysis()
        
        return report
    
    def print_summary(self):
        """
        Print a text summary of the dataset.
        """
        print("=" * 60)
        print("           📊 NETFLIX DATASET SUMMARY")
        print("=" * 60)
        print(f"\n📺 Total Content: {len(self.shows_df):,}")
        print(f"   • Movies: {len(self.shows_df[self.shows_df['type'] == 'Movie']):,}")
        print(f"   • TV Shows: {len(self.shows_df[self.shows_df['type'] == 'TV Show']):,}")
        
        print(f"\n📅 Year Range: {self.shows_df['release_year'].min()} - {self.shows_df['release_year'].max()}")
        
        print(f"\n🌍 Countries: {self.shows_df['country'].nunique()}")
        print(f"🎭 Unique Genres: {self.shows_df['listed_in'].nunique()}")
        
        if self.ratings_df is not None:
            sparsity = self.sparsity_analysis()
            print(f"\n👥 Users: {sparsity['n_users']:,}")
            print(f"⭐ Total Ratings: {sparsity['n_ratings']:,}")
            print(f"📉 Matrix Sparsity: {sparsity['sparsity_percent']}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    loader.load_netflix_data()
    loader.preprocess_shows()
    loader.generate_synthetic_ratings()
    
    # Run EDA
    eda = NetflixEDA(loader.shows_df, loader.ratings_df)
    eda.print_summary()
    
    # Generate report
    report = eda.generate_full_report()
    print(f"\n✅ Generated {len(report)} visualizations")
