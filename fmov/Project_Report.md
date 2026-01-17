📘 AI Project Report – Module E
	Student & Project Details
● Student Name :  Tarun
● Mentor Name  :  [Mentor Name]
● Project Title    :  Intelligent Content Recommendation System (fmov)


1. Problem Statement
	The rapid expansion of digital streaming services has led to an explosion of available content, creating a paradox of choice where users struggle to find movies or TV shows that match their specific mood or context. Traditional search platforms rely heavily on keyword matching or basic genre filters, which fail to capture the nuanced intent of natural language queries like "something short and funny for a bad day" or "scary movies about ghosts".
	This problem is highly relevant in the competitive streaming market, where user retention hinges on personalized discovery. Users expect intelligent systems that act as concierges rather than simple databases. The inability to process semantic meaning or visual cues (e.g., movie posters) in search queries often leads to user frustration and churn.
	The primary objectives of the system are to build a hybrid recommendation engine that understands natural language, analyzes visual inputs, and delivers personalized content suggestions through a modern, Netflix-style interface. The system aims to bridge the gap between human intent and metadata-based retrieval. It operates under constraints of using public datasets without real-time user history, assuming synthetic interactions for model training.

2. Approach
	- System Overview:
	The proposed solution, "fmov", is an active AI-powered content discovery platform. The workflow begins with the user interacting via a "Smart Chat" interface (text or image) or selecting mood-based filters. Input is routed through an intelligent controller that determines intent (Search vs. Recommendation). The system then queries a hybrid engine combining Collaborative Filtering and Content-Based logic to return ranked, personalized results displayed on a card-based dashboard.

	- Data Strategy:
	The system uses the "Netflix Movies and TV Shows" dataset from Kaggle (~8,800 records), containing attributes like title, description, cast, and release year. Data preprocessing steps included:
	    - Cleaning missing values in Director/Cast columns.
	    - Engineering a 'combined_features' text field for TF-IDF vectorization.
	    - Generating **Synthetic User Ratings** (500 users) with distinct cluster preferences (e.g., "Horror Fan") to enable the training of Collaborative Filtering models, as the original dataset lacked interaction data.

	- AI / Model Design:
	A hybrid ensemble approach is employed, combining:
		- **SVD (Singular Value Decomposition)**: For collaborative filtering to capture latent user preferences and provide serendipitous recommendations.
		- **TF-IDF & Cosine Similarity**: For content-based filtering to address the "Cold Start" problem and find similar items.
		- **Zero-Shot Classification (BART-MNLI)**: To function as an intent router, distinguishing between specific searches and general recommendation requests.
		- **CLIP (Vision Transformer)**: To analyze uploaded images (posters) and extract semantic tags for content matching.
	The system uses a "Smart Chat AI" controller that interprets queries in real-time. For example, "I'm bored" is classified as a recommendation request, triggering a mood-based search.

	- Tools & Technologies:
	    - **Python** for core logic.
	    - **Streamlit** for the web-based frontend.
	    - **Scikit-learn** for TF-IDF and metrics.
	    - **Surprise Library** for SVD Matrix Factorization.
	    - **Hugging Face Transformers** for BART, CLIP, and Sentence-BERT models.
	    - **Pandas/NumPy** for data manipulation.

3. Key Results
	The developed system successfully functions as a working prototype branded as "fmov", featuring a dark-themed, Netflix-like UI. Users can navigate via a top bar, browse a "Featured" hero section, and interact with the AI Chat.
	- **Observations**: The Zero-Shot classifier proved highly effective at handling ambiguous queries that traditional regex failed to catch. The Hybrid model demonstrated better subjective quality than isolated models, balancing popular hits (SVD) with relevant niche finds (Content-Based).
	- **Performance**: The model achieved an RMSE of ~1.38 on the synthetic test set, indicating reliable rating prediction. The application startup time was optimized from 45s to <5s using Lazy Loading for heavy AI models.
	- **Limitations**: The system relies on synthetic data for personalization, meaning real-world user adaption would require an initial data-gathering phase. The "Hero Section" occasionally faced data scarcity for specific years, requiring a fallback logic.

4. Learnings
	Through this project, I gained deep practical experience in building **Agentic AI** workflows, moving beyond simple model training to implementing intelligent controllers that route tasks. I learned how to integrate Multi-Modal AI (Vision + Text) into a cohesive web application.
	- **Technical Learnings**: Mastered the usage of `st.session_state` for managing navigation in Streamlit and realized the importance of `@st.cache_resource` for deploying heavy Transformer models efficiently.
	- **Challenges**: A key challenge was the "Hero Section" crashing due to empty dataframe filters, which taught me the importance of robust error handling and fallback mechanisms in production UI code.
	- **Future Improvements**: Future iterations would include Dockerizing the application for cloud deployment and implementing an online learning loop to update the SVD model in real-time as users rate content.

References & AI Usage Disclosure
● **Dataset**: Netflix Movies and TV Shows (Kaggle: shivamb/netflix-shows).
● **Tools**: Streamlit, Hugging Face Transformers (CLIP, BART), Scikit-learn, Surprise.
● **AI Usage**: This project was developed with the assistance of an AI Coding Agent (Google DeepMind) for code generation, debugging, and documentation structure. Synthetic datasets were procedurally generated using Python.
