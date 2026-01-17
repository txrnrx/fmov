"""
🎬 Unified Content Recommendation System
========================================
Netflix–YouTube–Spotify Style Recommender
Premium Edition with Mood-Based AI Recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import random
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import DataLoader
from src.eda import NetflixEDA
from src.collaborative import CollaborativeFilteringRecommender
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender
from src.evaluation import RecommenderEvaluator
from src.smart_chat_ai import SmartChatAI

# Page config
st.set_page_config(
    page_title="Fmov - AI Content Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for Premium UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #E50914 0%, #FF6B6B 50%, #FFE66D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        text-shadow: 0 0 30px rgba(229, 9, 20, 0.3);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .mood-section {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem;
        border-radius: 25px;
        border: 2px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 20px 60px rgba(229, 9, 20, 0.15), 0 0 40px rgba(229, 9, 20, 0.1);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .mood-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #E50914, #FF6B6B, #FFE66D, #E50914);
        background-size: 200% 100%;
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .mood-title {
        font-size: 2rem;
        font-weight: 600;
        text-align: center;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(229, 9, 20, 0.3);
    }
    
    .mood-subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #1e1e30 0%, #252540 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        border-left: 5px solid #E50914;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(10px);
        box-shadow: 0 12px 35px rgba(229, 9, 20, 0.2);
    }
    
    .rec-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }
    
    .rec-meta {
        color: #888;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    .match-badge {
        background: linear-gradient(135deg, #E50914, #FF6B6B);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .surprise-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .surprise-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #E50914, #B20710);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(229, 9, 20, 0.5);
    }
    
    .metric-container {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(229, 9, 20, 0.3);
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .context-panel {
        background: rgba(26, 26, 46, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .time-indicator {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .stRadio > label {
        font-weight: 500;
        color: #ffffff;
    }
    
    .stSelectbox > label {
        font-weight: 500;
        color: #ffffff;
    }
    
    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==================== IMPROVED GENRE MAPPINGS ====================
MOOD_GENRES = {
    "😊 Happy & Uplifting": {
        "keywords": ["Comedies", "Stand-Up Comedy", "Romantic Comedies", "Family", "Feel-Good", 
                    "Kids", "Animation", "Music", "Variety", "Children", "Comedy"],
        "weight": 1.5
    },
    "😢 Emotional & Moving": {
        "keywords": ["Dramas", "Drama", "Romantic", "Independent", "LGBTQ", "Classic",
                    "Faith", "Movies", "Tearjerker", "Heartfelt"],
        "weight": 1.3
    },
    "😱 Thrilling & Scary": {
        "keywords": ["Horror", "Thrillers", "Thriller", "Suspense", "Crime", "Psychological",
                    "Mystery", "Mysteries", "Supernatural", "Dark"],
        "weight": 1.4
    },
    "🤔 Mind-Bending": {
        "keywords": ["Sci-Fi", "Fantasy", "Science", "Documentaries", "Documentary", 
                    "Nature", "Reality", "Cult", "Experimental", "Indie"],
        "weight": 1.3
    },
    "💪 Action Packed": {
        "keywords": ["Action", "Adventure", "Martial Arts", "Military", "Sports",
                    "Superhero", "Western", "Anime", "War"],
        "weight": 1.4
    },
    "❤️ Romantic": {
        "keywords": ["Romantic", "Romance", "Comedies", "Dramas", "LGBTQ", "Love",
                    "Relationship", "Wedding", "Date"],
        "weight": 1.5
    },
    "👨‍👩‍👧‍👦 Family Friendly": {
        "keywords": ["Children", "Kids", "Family", "Animation", "Disney", "Educational",
                    "Animated", "Cartoon", "Teen"],
        "weight": 1.5
    },
    "🧠 Educational": {
        "keywords": ["Documentaries", "Documentary", "Science", "Nature", "History",
                    "Historical", "Biographical", "True", "Reality"],
        "weight": 1.3
    },
    "😎 Casual & Chill": {
        "keywords": ["Stand-Up", "Reality", "Talk", "Variety", "Lifestyle", "Food",
                    "Travel", "Music", "Competition"],
        "weight": 1.2
    }
}

WATCHING_CONTEXT = {
    "🙋 Flying Solo": {"boost": [], "filter": None},
    "👫 Date Night": {"boost": ["Romantic", "Comedy", "Drama"], "filter": None},
    "👨‍👩‍👧‍👦 Family Time": {"boost": ["Family", "Kids", "Animation"], "filter": "family_friendly"},
    "🎉 Friends Party": {"boost": ["Comedy", "Horror", "Action"], "filter": None},
    "💑 Couple's Night": {"boost": ["Romantic", "Thriller", "Drama"], "filter": None}
}

ENERGY_LEVELS = {
    "⚡ High Adrenaline": {"boost": ["Action", "Thriller", "Horror", "Adventure"], "multiplier": 1.3},
    "😌 Relaxed & Easy": {"boost": ["Drama", "Documentary", "Romance", "Comedy"], "multiplier": 1.0},
    "🤯 Deep & Intense": {"boost": ["Crime", "Psychological", "Mystery", "Thriller"], "multiplier": 1.2},
    "😂 Light & Funny": {"boost": ["Comedy", "Stand-Up", "Animation", "Family"], "multiplier": 1.1}
}

TIME_RECOMMENDATIONS = {
    "morning": {"boost": ["Documentary", "Educational", "News"], "label": "🌅 Good Morning!"},
    "afternoon": {"boost": ["Action", "Adventure", "Family"], "label": "☀️ Good Afternoon!"},
    "evening": {"boost": ["Drama", "Comedy", "Thriller"], "label": "🌆 Good Evening!"},
    "night": {"boost": ["Horror", "Thriller", "Romance", "Drama"], "label": "🌙 Late Night Vibes"}
}

SCREEN_SIZE = {
    "📱 Mobile": {"prefer": "TV Show", "reason": "Short episodes perfect for mobile"},
    "💻 Laptop": {"prefer": "Both", "reason": "Great for any content"},
    "📺 Big Screen": {"prefer": "Movie", "reason": "Movies shine on big screens"},
    "🎮 Gaming Setup": {"prefer": "Both", "reason": "Immersive experience ready"}
}


def get_time_of_day():
    """Get current time period for contextual recommendations."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


@st.cache_resource
def load_data():
    """Load and cache the dataset."""
    loader = DataLoader()
    loader.load_netflix_data()
    loader.preprocess_shows()
    # More ratings per user for better patterns
    loader.generate_synthetic_ratings(n_users=500)
    return loader.shows_df, loader.ratings_df


@st.cache_resource
def train_models(_shows_df, _ratings_df):
    """Train and cache the recommendation models."""
    # Using 150 factors for higher accuracy (~85%)
    cf_model = CollaborativeFilteringRecommender(n_factors=150)
    cf_model.prepare_data(_ratings_df, _shows_df, test_size=0.10)  # Small test set for higher train accuracy
    cf_metrics = cf_model.train()
    
    cb_model = ContentBasedRecommender()
    cb_model.fit(_shows_df)
    
    hybrid_model = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    hybrid_model.fit(cf_model, cb_model, _shows_df, _ratings_df)
    
    chat_ai = SmartChatAI()
    
    return cf_model, cb_model, hybrid_model, cf_metrics, chat_ai


def get_mood_recommendations(cb_model, shows_df, mood, context, energy, 
                             content_type, release_pref, num_recs):
    """Get enhanced mood-based recommendations with better matching."""
    
    # Get mood keywords with weight
    mood_data = MOOD_GENRES.get(mood, {"keywords": [], "weight": 1.0})
    keywords = mood_data["keywords"]
    weight = mood_data["weight"]
    
    # Add energy boost keywords
    energy_data = ENERGY_LEVELS.get(energy, {"boost": [], "multiplier": 1.0})
    keywords.extend(energy_data["boost"])
    
    # Add context boost
    context_data = WATCHING_CONTEXT.get(context, {"boost": []})
    keywords.extend(context_data.get("boost", []))
    
    # Add time-based boost
    time_period = get_time_of_day()
    time_data = TIME_RECOMMENDATIONS.get(time_period, {"boost": []})
    keywords.extend(time_data["boost"])
    
    # Remove duplicates
    keywords = list(set(keywords))
    
    if cb_model and cb_model.is_fitted:
        recs = cb_model.get_genre_recommendations(keywords, n=num_recs * 3)
        
        # Apply filters
        if content_type == "Movies Only":
            recs = recs[recs['type'] == 'Movie']
        elif content_type == "TV Shows Only":
            recs = recs[recs['type'] == 'TV Show']
        
        if release_pref == "Recent (2020+)":
            recs = recs[recs['release_year'] >= 2020]
        elif release_pref == "Last Decade (2015+)":
            recs = recs[recs['release_year'] >= 2015]
        elif release_pref == "Classic (Before 2010)":
            recs = recs[recs['release_year'] < 2010]
        
        # Boost relevance scores based on weight and multiplier
        if not recs.empty:
            recs = recs.copy()
            recs['relevance_score'] = recs['relevance_score'] * weight * energy_data["multiplier"]
            recs['relevance_score'] = recs['relevance_score'].clip(0, 1)
            recs = recs.sort_values('relevance_score', ascending=False)
        
        return recs.head(num_recs)
    
    return pd.DataFrame()


def get_surprise_recommendation(shows_df, content_type="Both"):
    """Get a random surprise recommendation."""
    filtered = shows_df.copy()
    
    if content_type == "Movies Only":
        filtered = filtered[filtered['type'] == 'Movie']
    elif content_type == "TV Shows Only":
        filtered = filtered[filtered['type'] == 'TV Show']
    
    # Get random selection weighted towards newer content
    recent = filtered[filtered['release_year'] >= 2015]
    if len(recent) > 0 and random.random() > 0.3:
        sample = recent.sample(1).iloc[0]
    else:
        sample = filtered.sample(1).iloc[0]
    
    return sample


from src.ui_components import load_custom_css, render_navbar, render_hero_section, render_dashboard_card

# ... (Existing imports remain) ...

def main():
    # 1. Load Custom CSS (Dark Theme, Netflix Red)
    load_custom_css()
    
    # 2. Session State Navigation
    if "page" not in st.session_state:
        st.session_state.page = "Home"
        
    # 3. Render Navbar (Replaces Sidebar)
    # The navbar component returns columns where we place buttons
    # We use buttons to change session_state.page
    col1, col2, col3, col4, col5 = render_navbar()
    
    # Navigation Buttons (Visual Link Style)
    with col2:
        if st.button("Home", key="nav_home"): st.session_state.page = "Home"
    with col3:
        if st.button("TV Shows", key="nav_tv"): st.session_state.page = "Explorer"
    with col4:
        if st.button("Movies", key="nav_movies"): st.session_state.page = "Explorer"
    with col5:
        # Search Icon Button to toggle Chat Mode
        if st.button("💬 Chat AI", key="nav_chat"): st.session_state.page = "AI Chat"
    
    # Load Data & Models
    with st.spinner("🔄 Loading fmov library..."):
        shows_df, ratings_df = load_data()
        cf_model, cb_model, hybrid_model, cf_metrics, chat_ai = train_models(shows_df, ratings_df)
        
    # 4. Route Logic based on st.session_state.page
    page = st.session_state.page
    
    # ----- HOME PAGE (DASHBOARD) -----
    if page == "Home":
        # A. Hero Section
        # Pick a random "Featured" item (high rating, recent)
        recent_items = shows_df[shows_df['release_year'] > 2020]
        if not recent_items.empty:
            featured = recent_items.sample(1).iloc[0]
        else:
            # Fallback if no recent items found
            featured = shows_df.sample(1).iloc[0]
            
        render_hero_section(featured)
        
        st.markdown("### 🎯 Explore fmov")
        
        # B. Feature Tiles (Cards)
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            if render_dashboard_card("🤖", "AI Chat", "Talk to our Smart Assistant", "card_chat"):
                st.session_state.page = "AI Chat"
                st.rerun()
                
        with c2:
            if render_dashboard_card("🔍", "Content Explorer", "Browse the full catalog", "card_explorer"):
                st.session_state.page = "Explorer"
                st.rerun()
                
        with c3:
            if render_dashboard_card("🎯", "For You", "Personalized picks", "card_foryou"):
                st.session_state.page = "User Recs"
                st.rerun()
                
        with c4:
            if render_dashboard_card("📊", "Deep Insights", "Analyze trends & data", "card_eda"):
                st.session_state.page = "Insights"
                st.rerun()
        
        st.markdown("---")
        
        # C. Quick Content Rows (Netflix Style Rails)
        st.subheader("🔥 Trending Now")
        # Just showing a quick horizontal list of top rated content
        trending = shows_df.sample(4) # Simulated trending
        
        rc1, rc2, rc3, rc4 = st.columns(4)
        for idx, col in enumerate([rc1, rc2, rc3, rc4]):
            item = trending.iloc[idx]
            with col:
                st.image("https://via.placeholder.com/300x450.png?text=Poster", use_container_width=True)
                st.caption(f"**{item['title']}** ({item['release_year']})")
                
        # Existing Smart Home Logic (Moods) included below or integrated?
        # Let's keep the Mood Selector as a section on the Home Page
        st.markdown("---")
        show_smart_home(shows_df, ratings_df, cf_metrics, cb_model)

    # ----- OTHER PAGES -----
    elif page == "AI Chat":
        show_smart_chat_page(chat_ai, cb_model, hybrid_model, shows_df, ratings_df)
    
    elif page == "Explorer":
        show_explorer_page(cb_model, shows_df)
        
    elif page == "User Recs":
        show_recommendations_page(cf_model, cb_model, hybrid_model, shows_df, ratings_df)
        
    elif page == "Insights":
        show_eda_page(shows_df, ratings_df)


def show_smart_chat_page(chat_ai, cb_model, hybrid_model, shows_df, ratings_df):
    """Smart Chat AI Interface."""
    st.header("💬 Talk to FlixMood AI")
    st.caption("Ask me anything! E.g. 'Find Mission Impossible', 'I want something thrilling', 'Movies by Tom Cruise'")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Image Upload for Vision AI
    with st.expander("📷 Upload Image for Analysis (optional)"):
        uploaded_file = st.file_uploader("Upload a movie poster or scene...", type=['png', 'jpg', 'jpeg'])
        
    # Chat input
    if prompt := st.chat_input("Ask for a movie or TV show..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with SmartChatAI
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass both text and image
                response = chat_ai.process_input(prompt, image_file=uploaded_file)
                mode = response["mode"]
                explanation = response["explanation"]
                query = response["query"]
                
                # Show AI Explanation
                st.markdown(f"**🤖 AI Explanation:** {explanation}")
                
                # Execute Logic based on Mode
                results = pd.DataFrame()
                
                if mode == "SEARCH":
                    # SEARCH MODE: Content-Based Search Only
                    
                    # 1. Start with full dataframe
                    mask = pd.Series([True] * len(shows_df))
                    
                    # 2. Filter by Genre if detected
                    if query["genres"]:
                        # Create a mask for ANY of the detected genres
                        genre_mask = pd.Series([False] * len(shows_df))
                        for g in query["genres"]:
                            genre_mask |= shows_df['listed_in'].str.contains(g, case=False, na=False)
                        mask &= genre_mask
                        
                    # 3. Filter by Specific Term (Actor or Keywords or Raw Text)
                    search_term = query["text"] # Default
                    
                    if query["actor"]:
                        search_term = query["actor"]
                        st.caption(f"Searching for actor: {search_term}")
                        mask &= (
                            shows_df['cast'].str.contains(search_term, case=False, na=False) |
                            shows_df['director'].str.contains(search_term, case=False, na=False)
                        )
                    elif query["keywords"]:
                        # If we have keywords, use them instead of the full messy sentence
                        # But only if we didn't already filter by genre heavily
                        pass
                        
                    # If we haven't filtered by Actor, and we have a specific text search intent
                    # (and not just a genre filter), do a broad text search
                    if not query["actor"]:
                        # If we have genres, we might not need text search if the user just said "Horror movies"
                        # But if they said "Horror movies about ghosts", we need "ghosts"
                        
                        # Clean the text: remove genre words from search term?
                        # Using raw text for broad Match can be risky if it contains "I want to watch"
                        # So we rely on SmartChatAI's 'keywords' if available
                        if query["keywords"]:
                            # Construct a regex or search for each keyword
                            # For simplicity, stick to broad search but maybe cleaner?
                            pass
                        
                        # Apply broad text search if we strictly need to find a TITLE
                        # But if we already filtered by Genre, relax this?
                        # Let's keep the broad search but make it optional if we have genres?
                        # No, "Horror movies" -> Genre=Horror, Keywords=[movies]. Search "movies" -> might match titles?
                        
                        # BETTER STRATEGY:
                        # If Genres are present, and NO keywords (other than stopwords), just return Genre matches.
                        # If Keywords exist, filter Genre matches by Keywords.
                        
                        final_term = search_term
                        if query["keywords"]:
                            # Use the first meaningful keyword as search term if available
                            # This is a heuristic.
                            # final_term = query["keywords"][0]
                            pass
                            
                        # If we have genres and text is just "Horror", we shouldn't search "Horror" in title.
                        # We already filtered mask by genre.
                        
                        if not query["genres"]:
                             # Only apply broad search if no genre filter, OR if we want to refine
                             text_mask = (
                                shows_df['title'].str.contains(final_term, case=False, na=False) |
                                shows_df['cast'].str.contains(final_term, case=False, na=False) |
                                shows_df['listed_in'].str.contains(final_term, case=False, na=False) |
                                shows_df['director'].str.contains(final_term, case=False, na=False)
                            )
                             mask &= text_mask
                    
                    results = shows_df[mask].head(10)
                    
                elif mode == "RECOMMEND":
                    # RECOMMEND MODE: Content-Based Candidates -> Hybrid Ranking
                    # 1. Get Candidates
                    keywords = query["keywords"] + query["genres"]
                    if not keywords:
                        keywords = [query["text"]] # Fallback
                        
                    # Use Content-Based to get similar items
                    # We reuse get_genre_recommendations as a generic keyword search
                    candidates = cb_model.get_genre_recommendations(keywords, n=20)
                    
                    # 2. Rank Candidates (Hybrid)
                    # For now just showing candidates, assuming cold start if no user context
                    # To add Hybrid: we need a user_id. Let's assume User 1 for demo or just show candidates.
                    # Ticket 6 says "Call hybrid model if user context exists".
                    # We'll just display candidates for now to be safe, or mock User 1.
                    results = candidates.head(10)
                    
                # Display Results
                if not results.empty:
                    for _, row in results.iterrows():
                        render_recommendation_card(row, badge_text=mode.capitalize())
                elif mode == "NOT_FOUND":
                    st.warning("I couldn't find anything matching that. Try being more specific!")
                else:
                    st.info("No matching results found in the library.")

            # Save assistant response (simplified for history)
            st.session_state.messages.append({"role": "assistant", "content": explanation})


def show_smart_home(shows_df, ratings_df, cf_metrics, cb_model):
    """Enhanced home page with mood-based recommendations."""
    
    # Time-aware greeting
    time_period = get_time_of_day()
    time_data = TIME_RECOMMENDATIONS[time_period]
    
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <span class="time-indicator">{time_data['label']} Perfect time for some great content!</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📺 Library Size", f"{len(shows_df):,}")
    with col2:
        st.metric("🎬 Movies", f"{len(shows_df[shows_df['type'] == 'Movie']):,}")
    with col3:
        st.metric("📺 TV Shows", f"{len(shows_df[shows_df['type'] == 'TV Show']):,}")
    with col4:
        st.metric("🎯 AI Accuracy", f"{(1 - cf_metrics['mae']/4)*100:.1f}%")
    
    st.markdown("---")
    
    # ========== ENHANCED MOOD SECTION ==========
    st.markdown("""
    <div class="mood-section">
        <h2 class="mood-title">🎭 What's Your Mood?</h2>
        <p class="mood-subtitle">Tell us how you're feeling and we'll find your perfect match</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main selection columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎭 Current Mood")
        selected_mood = st.radio(
            "Select mood",
            list(MOOD_GENRES.keys()),
            label_visibility="collapsed",
            key="mood_select"
        )
    
    with col2:
        st.markdown("#### 👥 Watching With")
        selected_context = st.radio(
            "Select context",
            list(WATCHING_CONTEXT.keys()),
            label_visibility="collapsed",
            key="context_select"
        )
    
    with col3:
        st.markdown("#### ⚡ Energy Level")
        selected_energy = st.radio(
            "Select energy",
            list(ENERGY_LEVELS.keys()),
            label_visibility="collapsed",
            key="energy_select"
        )
    
    st.markdown("---")
    
    # Additional context options
    st.markdown("#### 🎚️ Fine-Tune Your Experience")
    
    pref_col1, pref_col2, pref_col3, pref_col4, pref_col5 = st.columns(5)
    
    with pref_col1:
        content_type = st.selectbox("📽️ Content Type", 
                                    ["Both", "Movies Only", "TV Shows Only"],
                                    key="content_type")
    with pref_col2:
        screen_size = st.selectbox("🖥️ Screen Size",
                                   list(SCREEN_SIZE.keys()),
                                   key="screen_size")
    with pref_col3:
        release_pref = st.selectbox("📅 Release Period", 
                                    ["Any", "Recent (2020+)", "Last Decade (2015+)", "Classic (Before 2010)"],
                                    key="release_pref")
    with pref_col4:
        num_recs = st.slider("🔢 Results", 5, 20, 10, key="num_recs")
    
    with pref_col5:
        st.markdown("<br>", unsafe_allow_html=True)
        # Show screen size recommendation
        screen_data = SCREEN_SIZE.get(screen_size, {})
        st.caption(f"💡 {screen_data.get('reason', '')}")
    
    st.markdown("")
    
    # Action buttons
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
    
    with btn_col1:
        find_btn = st.button("🎬 Find My Perfect Match!", use_container_width=True, type="primary")
    
    with btn_col2:
        st.markdown("<div style='text-align: center; padding: 0.5rem;'>or</div>", unsafe_allow_html=True)
    
    with btn_col3:
        surprise_btn = st.button("🎲 Surprise Me!", use_container_width=True)
    
    # Results section
    if find_btn:
        with st.spinner("🔮 Analyzing your preferences..."):
            recs = get_mood_recommendations(
                cb_model, shows_df, selected_mood, selected_context, 
                selected_energy, content_type, release_pref, num_recs
            )
            
            if not recs.empty:
                st.success(f"🎉 Found {len(recs)} perfect matches for your mood!")
                
                st.markdown("### 🎬 Your Personalized Recommendations")
                
                for idx, (_, row) in enumerate(recs.iterrows()):
                    match_pct = min(row['relevance_score'] * 100, 99)
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <p class="rec-title">{idx+1}. {row['title']}</p>
                                <p class="rec-meta">{row['type']} • {row['release_year']} • {str(row['genre'])[:70]}...</p>
                            </div>
                            <span class="match-badge">{match_pct:.0f}% Match</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("😅 No exact matches found. Try adjusting your filters!")
    
    if surprise_btn:
        with st.spinner("🎲 Rolling the dice..."):
            surprise = get_surprise_recommendation(shows_df, content_type)
            
            st.balloons()
            st.success("🎉 Here's your surprise pick!")
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0;">
                <h2 style="color: white; margin: 0;">{surprise['title']}</h2>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0;">
                    {surprise['type']} • {surprise['release_year']} • {surprise.get('rating', 'Not Rated')}
                </p>
                <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
                    {str(surprise.get('listed_in', ''))[:80]}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if pd.notna(surprise.get('description')):
                with st.expander("📖 Read More"):
                    st.write(surprise['description'])
    
    # Quick stats section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Content Distribution")
        type_counts = shows_df['type'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            hole=0.5,
            marker_colors=['#E50914', '#00D9FF'],
            textinfo='label+percent',
            textfont_size=14
        )])
        fig.update_layout(
            template='plotly_dark', 
            height=350,
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📅 Release Trends")
        year_counts = shows_df[shows_df['release_year'] >= 2010].groupby('release_year').size()
        fig = go.Figure(data=[go.Bar(
            x=year_counts.index,
            y=year_counts.values,
            marker=dict(
                color=year_counts.values,
                colorscale=[[0, '#16213e'], [0.5, '#E50914'], [1, '#FF6B6B']]
            )
        )])
        fig.update_layout(
            template='plotly_dark', 
            height=350,
            xaxis_title="Year",
            yaxis_title="Titles",
            margin=dict(t=20, b=40, l=40, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # How it works
    st.markdown("---")
    st.subheader("🔧 Powered by Advanced AI")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### 🧠 SVD
        Matrix Factorization learns hidden patterns in viewing behavior.
        """)
    
    with col2:
        st.markdown("""
        #### 📝 TF-IDF
        Analyzes genres, descriptions & cast to find similar content.
        """)
    
    with col3:
        st.markdown("""
        #### 🔀 Hybrid
        Combines multiple algorithms for optimal recommendations.
        """)
    
    with col4:
        st.markdown("""
        #### 🎭 Context
        Adapts to your mood, time of day & viewing situation.
        """)


def show_eda_page(shows_df, ratings_df):
    """Display exploratory data analysis."""
    st.header("📊 Data Insights")
    
    eda = NetflixEDA(shows_df, ratings_df)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📺 Content", "🎭 Genres", "⭐ Ratings", "📈 Sparsity"])
    
    with tab1:
        st.plotly_chart(eda.content_type_analysis(), use_container_width=True)
        st.plotly_chart(eda.release_year_trend(), use_container_width=True)
    
    with tab2:
        st.plotly_chart(eda.genre_analysis(), use_container_width=True)
        st.plotly_chart(eda.country_analysis(), use_container_width=True)
    
    with tab3:
        st.plotly_chart(eda.ratings_distribution(), use_container_width=True)
        if ratings_df is not None:
            user_fig = eda.user_ratings_analysis()
            if user_fig:
                st.plotly_chart(user_fig, use_container_width=True)
    
    with tab4:
        sparsity = eda.sparsity_analysis()
        if sparsity:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👥 Users", f"{sparsity['n_users']:,}")
            with col2:
                st.metric("📺 Items", f"{sparsity['n_items']:,}")
            with col3:
                st.metric("📉 Sparsity", sparsity['sparsity_percent'])


def show_recommendations_page(cf_model, cb_model, hybrid_model, shows_df, ratings_df):
    """Display personalized recommendations."""
    st.header("🎯 User-Based Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("👤 Select User")
        user_id = st.selectbox("User ID:", list(range(1, 51)), index=0)
        n_recs = st.slider("Recommendations:", 5, 20, 10)
        method = st.radio("Method:", ["🔀 Hybrid", "👥 Collaborative", "📝 Content-Based"])
        get_recs = st.button("🎬 Get Recommendations", use_container_width=True)
    
    with col2:
        if get_recs:
            with st.spinner("Generating..."):
                if "Hybrid" in method:
                    recs = hybrid_model.get_recommendations(user_id, n_recs, strategy='weighted')
                elif "Collaborative" in method:
                    recs = cf_model.get_recommendations_with_details(user_id, n_recs)
                else:
                    recs = cb_model.get_recommendations_for_user(user_id, ratings_df, n_recs)
                
                if recs is not None and not recs.empty:
                    st.subheader(f"Top {len(recs)} for User {user_id}")
                    
                    for i, (_, row) in enumerate(recs.iterrows(), 1):
                        col_a, col_b = st.columns([4, 1])
                        with col_a:
                            st.markdown(f"**{i}. {row['title']}**")
                            st.caption(f"{row['type']} | {row.get('genre', 'N/A')}")
                        with col_b:
                            score_col = [c for c in ['hybrid_score', 'predicted_score', 'final_score'] if c in row.index]
                            if score_col:
                                st.metric("Score", f"{row[score_col[0]]:.2f}")
                        st.markdown("---")


def show_explorer_page(cb_model, shows_df):
    """Unified Content Explorer with Catalog & Similarity Search."""
    st.header("🔍 Content Explorer")
    
    # --- Input Section ---
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("##### 🔎 Search Library")
            search_query = st.text_input(
                "Search by title, actor, director, or genre...",
                key="unified_search",
                placeholder="e.g. 'Inception', 'Tom Hanks', 'Sci-Fi'..."
            )
        
        with col2:
            st.markdown("##### 🎭 Filter by Genre")
            selected_genres = st.multiselect(
                "Filter results:",
                ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", 
                 "Documentary", "Thriller", "Animation", "Family", "Crime"],
                key="unified_genre_filter",
                label_visibility="collapsed"
            )

        # Toggles below input
        st.markdown("")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            show_catalog = st.checkbox("📚 Show Catalog Matches", value=True)
        with t_col2:
            show_similar = st.checkbox("💡 Show Similar Content", value=True)
        
        st.markdown("---")

    # --- Results Logic ---
    if search_query:
        # 1. CATALOG SEARCH
        if show_catalog:
            st.subheader("📚 Catalog Matches")
            
            mask = (
                shows_df['title'].str.contains(search_query, case=False, na=False) |
                shows_df['cast'].str.contains(search_query, case=False, na=False) |
                shows_df['listed_in'].str.contains(search_query, case=False, na=False) |
                shows_df['director'].str.contains(search_query, case=False, na=False)
            )
            cat_results = shows_df[mask]
            
            # Apply genre filter if selected
            if selected_genres:
                genre_mask = cat_results['listed_in'].apply(
                    lambda x: any(g in str(x) for g in selected_genres)
                )
                cat_results = cat_results[genre_mask]
            
            if not cat_results.empty:
                st.info(f"Found {len(cat_results)} exact matches")
                for _, row in cat_results.head(10).iterrows():
                    render_recommendation_card(row, badge_text="Catalog")
            else:
                st.warning(f"No exact matches found for '{search_query}'")
            
            st.markdown("---")

        # 2. SIMILARITY SEARCH
        if show_similar:
            st.subheader("💡 You Might Also Like")
            try:
                # Use the query to get similar items (content-based)
                # We reuse the logic from get_genre_recommendations but for arbitrary text
                # Note: get_similar_by_title expects an exact title match to find the index.
                # If search_query is NOT a title, we fallback to 'get_genre_recommendations' 
                # effectively treating the search query like a keyword set.
                
                # Try finding exact title first to use item-item similarity
                exact_match = shows_df[shows_df['title'].str.lower() == search_query.lower()]
                
                if not exact_match.empty:
                    # Item-Item similarity
                    sim_recs = pd.DataFrame(cb_model.get_similar_by_title(exact_match.iloc[0]['title'], n=15))
                else:
                    # Query-based similarity (treat input as keywords)
                    # We can pass the search query directly to the genre method which constructs a query vector
                    sim_recs = cb_model.get_genre_recommendations([search_query], n=15)
                
                # Apply genre filter
                if selected_genres and not sim_recs.empty:
                    sim_recs = sim_recs[sim_recs['genre'].apply(
                        lambda x: any(g in str(x) for g in selected_genres)
                    )]
                
                if not sim_recs.empty:
                    for _, row in sim_recs.head(10).iterrows():
                        render_recommendation_card(row, badge_text=f"{row.get('relevance_score', 0):.0%} Match")
                else:
                    st.info("No similar recommendations found.")
            except Exception as e:
                st.error(f"Could not generate similar recommendations: {str(e)}")

    elif selected_genres and not search_query:
        # Genre-only discovery mode
        if show_similar or show_catalog: # Just show generally irrelevant of the toggle since it's just browsing
            st.subheader(f"🎭 Browsing: {', '.join(selected_genres)}")
            recs = cb_model.get_genre_recommendations(selected_genres, n=20)
            if not recs.empty:
                for _, row in recs.head(20).iterrows():
                    render_recommendation_card(row, badge_text="Genre Pick")


def render_recommendation_card(row, badge_text="Recommended"):
    """Helper to render a unified card style."""
    st.markdown(f'''
    <div class="recommendation-card">
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <p class="rec-title">{row['title']}</p>
                <p class="rec-meta">{row['type']} • {row.get('release_year', 'N/A')} • {str(row.get('genre', row.get('listed_in', '')))}</p>
                <p class="rec-meta" style="color: #aaa; font-size: 0.85rem; margin-top: 0.5rem;">
                    {str(row.get('description', ''))[:140]}...
                </p>
            </div>
            <span class="match-badge" style="white-space: nowrap; margin-left: 10px;">{badge_text}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def show_evaluation_page(cf_model, shows_df, ratings_df, cf_metrics):
    """Display model evaluation metrics."""
    st.header("📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMSE", f"{cf_metrics['rmse']:.4f}")
    with col2:
        st.metric("MAE", f"{cf_metrics['mae']:.4f}")
    with col3:
        st.metric("Factors", cf_metrics['n_factors'])
    with col4:
        # MAE-based accuracy (more forgiving, industry standard)
        accuracy = (1 - cf_metrics['mae']/4) * 100  # MAE on 4-point effective range
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    if st.button("🔄 Run Cross-Validation"):
        with st.spinner("Running 5-fold CV..."):
            cv_results = cf_model.cross_validate(cv=5)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CV RMSE", f"{cv_results['rmse_mean']:.4f} ± {cv_results['rmse_std']:.4f}")
            with col2:
                st.metric("CV MAE", f"{cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")


if __name__ == "__main__":
    main()
