import streamlit as st

def load_custom_css():
    """Inject custom CSS for the Netflix-style (fmov) theme."""
    st.markdown("""
        <style>
        /* 1. Main Background & Text */
        .stApp {
            background-color: #141414;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }
        
        h1, h2, h3, h4, 5, p, span, div {
            color: #e5e5e5;
        }
        
        /* 2. Brand Colors */
        :root {
            --primary-red: #E50914;
            --dark-bg: #141414;
            --card-bg: #181818;
        }

        /* 3. Hide Default Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background-color: transparent; /* Transparent header for immersive feel */
        }
        
        /* 4. Custom Navbar Styling (Placeholder for visual structure) */
        .fmov-logo {
            color: #E50914;
            font-size: 2.5rem;
            font-weight: 800;
            text-shadow: 0px 1px 2px rgba(0,0,0,0.8);
            margin-bottom: 0px;
            padding-bottom: 0px;
            cursor: pointer;
        }
        
        /* 5. Feature Cards (Dashboard) */
        div[data-testid="stMetric"], div.stButton > button {
            background-color: transparent; 
        }

        /* Custom Card Container */
        .feature-card {
            background-color: #181818;
            border-radius: 4px;
            padding: 20px;
            border: 1px solid #333;
            transition: transform 0.2s ease, border-color 0.2s;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            cursor: pointer;
        }
        
        .feature-card:hover {
            transform: scale(1.03);
            border-color: #E50914;
            background-color: #1f1f1f;
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 10px;
        }
        
        .feature-title {
            font-size: 1.2rem;
            font-weight: bold;
            color: white;
            margin-bottom: 5px;
        }
        
        .feature-desc {
            font-size: 0.85rem;
            color: #b3b3b3;
        }

        /* 6. Streamlit Widgets Overrides */
        /* Buttons */
        .stButton > button {
            background-color: #E50914;
            color: white;
            border: none;
            border-radius: 3px;
            font-weight: bold;
            padding: 0.5rem 1rem;
            transition: background-color 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #b20710; /* Darker red on hover */
            color: white;
            border: none;
        }
        
        /* Inputs */
        div[data-baseweb="input"] {
            background-color: #333;
            border: 1px solid #555;
            border-radius: 2px;
            color: white;
        }
        
        input[class="st-az"] {
            color: white;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #2b2b2b;
            color: #fff;
            font-weight: bold;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            color: #b3b3b3;
            border: none;
            font-size: 1rem;
            padding-bottom: 10px;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #fff;
            border-bottom: 3px solid #E50914;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

def render_navbar():
    """Renders the top navigation bar."""
    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 3.5])
    
    with col1:
        st.markdown('<div class="fmov-logo">fmov</div>', unsafe_allow_html=True)
    
    # We return the columns so the main app can place buttons inside them 
    # (since standard HTML links won't update Streamlit session state)
    return col1, col2, col3, col4, col5

def render_hero_section(featured_movie):
    """Renders a Hero Banner for a featured item."""
    # Assuming featured_movie is a dict or Series row
    title = featured_movie.get('title', 'Featured Title')
    desc = featured_movie.get('description', 'An amazing movie you must watch.')
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #141414 10%, transparent 60%),
            linear-gradient(180deg, transparent 0%, #141414 100%), 
            url('https://images.unsplash.com/photo-1594909122845-11baa439b7bf?auto=format&fit=crop&w=1200&q=80');
            background-size: cover; 
            background-position: center;
            padding: 5rem 2rem 3rem 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
        ">
            <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px #000;">{title}</h1>
            <p style="font-size: 1.1rem; max-width: 600px; text-shadow: 1px 1px 2px #000; margin-bottom: 1.5rem;">{desc}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Place buttons below the CSS background (Streamlit limitation: buttons can't be inside HTML easily)
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("▶ Play", key="hero_play"):
            st.toast("Playing movie... (Demo)")
    with c2:
        if st.button("ℹ More Info", key="hero_info"):
            st.toast(f"Showing info for {title}")

def render_dashboard_card(icon, title, desc, key_name):
    """Renders a clickable card component."""
    # We use a button that spans the full width/height appearance via CSS
    # But explicitly, Streamlit buttons are simple.
    # To make it look like a card, we can put a button inside a container or just use the button with custom class.
    
    # Alternative: Use HTML for visual, button for action? Hard to overlay.
    # Approach: Use a container with markdown styles, and a button below/inside.
    
    st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Invisible full-width button overlay logic is hard in pure Streamlit.
    # We will just put a distinct button below it or rely on the user clicking the button 
    # that we place right after.
    return st.button(f"Go to {title}", key=key_name, use_container_width=True)
