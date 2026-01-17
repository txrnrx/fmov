# fmov: AI-Powered Content Recommendation System
**Comprehensive Presentation Guide (10 Slides)**

This document serves as both the **Slide Content** and the **Detailed Speaker Script** for your presentation. Each slide includes the visual points to display and a deep-dive explanation for you to speak or include in your report.

---

## Slide 1: Title Slide

### **Visual Content**
*   **Project Title**: **fmov**: Intelligent Content Recommendation System
*   **Subtitle**: A Hybrid AI Approach with Agentic Chat & Vision Analysis
*   **Student Name**: Tarun
*   **Mentor Name**: [Mentor Name]
*   **Visual**: A high-quality screenshot of the "fmov" Home Dashboard, showing the dark theme, the "Hero" banner of a featured movie, and the "Smart Chat" floating button.

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"Good morning everyone. Today I am presenting **'fmov'**, an intelligent Content Recommendation System designed to tackle the specific challenges of the modern streaming era.
This project is not just a database of movies; it is a full-stack AI application that mimics the user experience of platforms like Netflix while introducing novel 'Agentic' capabilities.
When we started this project, our goal was simple: move beyond static keyword search. If you look at the interface on the screen, you'll see it strongly resembles a premium OTT platform. We call it **'fmov'**.
Why this name? It stands for 'Future Movies' or 'Fast Movies' - hinting at the speed and intelligence of discovery.
The core innovation I'll be discussing today is the **Hybrid Engine**. Most student projects use either Collaborative Filtering (which fails for new users) or Content-Based Filtering (which traps users in a bubble). 'fmov' uses both, orchestrated by a **Smart Chat AI**.
This system integrates **Computer Vision** (OpenAI CLIP) to understand movie posters, **Zero-Shot Classification** (BART) to understand user intent, and **SVD Matrix Factorization** to predict user ratings.
Throughout this presentation, I will walk you through the real-world problem of 'Choice Paralysis', the System Architecture we designed to solve it, a live demonstration of our 'Smart Chat' feature, and the quantitative metrics that prove our model's accuracy.
What you see here on the title slide is the actual working prototype, running on Streamlit with a custom CSS framework to achieve that 'Dark Mode' aesthetic.
This isn't just a backend script; it's a complete product experience. The 'Hero Section' you see dynamically updates based on the catalog, and the navigation is fully state-aware.
By the end of this talk, I hope to demonstrate how Large Language Models and traditional Recommender Systems can work together to create a 'Concierge' experience rather than just a 'Search Bar'.
Let's dive into the problem we are solving."

---

## Slide 2: Problem Overview

### **Visual Content**
*   **The Paradox of Choice**:
    *   Netflix has ~15,000 titles.
    *   Avg user spends **18 minutes** searching per session.
*   **Limitations of Current Systems**:
    *   ❌ **Rigid Keyword Search**: "Action" works, but "Heartwarming story for a rainy day" fails.
    *   ❌ **No Visual Context**: Can't search by "movies that look like this".
    *   ❌ **Cold Start**: New users get generic/random suggestions.
*   **Objective**: Build a system that understands **Nuance**, **Context**, and **Semantics**.

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"Let's identify the core problem using the 'Paradox of Choice'. As streaming libraries grow exponentially—Netflix alone adds thousands of hours of content annually—discovery becomes broken.
Studies show the average user spends nearly 20 minutes just scrolling through thumbnails before picking something to watch. Often, they give up and don't watch anything. This is 'Churn', and it's a billion-dollar problem for streaming services.
Why does this happen? Because traditional search engines are **Rigid**.
If I type 'Action Movies', the system essentially does a `SELECT * FROM movies WHERE genre='Action'`. That's easy.
But humans don't always think in genres. We think in **Moods** and **Contexts**.
If I say, 'I had a bad day, show me something short and funny,' a traditional SQL-based search fails completely. It looks for the keyword 'bad day' in the title and finds nothing.
Furthermore, most systems are text-only. If I see a cool poster on Instagram and want to find that movie, I can't just upload it to Netflix.
Another major issue is the **Cold Start Problem**. When a new user joins, the system knows nothing about them. Collaborative Filtering fails here because there's no history vector to compare. So, platforms just show 'Trending Now', which isn't personalized.
The objective of 'fmov' is to bridge this gap. We wanted to build an AI that acts less like a database and more like a **Concierge**.
A concierge doesn't just ask 'What Genre?'. They ask 'How are you feeling?'. They look at standard metadata but also understand the *vibe* of a request.
We aim to solve three specific friction points:
1.  **Semantic Gap**: Bridging the gap between 'I want something scary' and the metadata tag 'Horror'.
2.  **Visual Gap**: allowing users to search using images.
3.  **Context Gap**: Providing different recommendations for 'Date Night' vs. 'Family Sunday'.
This requires a shift from simple Metadata Filtering to **Agentic AI** that can reason about intent. That is the core mission of this project."

---

## Slide 3: Solution Overview ("fmov")

### **Visual Content**
*   **The "fmov" Platform**: A 3-Pillar Approach.
    1.  **Netflix-Style UI** (Frontend):
        *   Dark Theme, Responsive Cards, Top Navigation.
        *   "Hero" Section for featured content.
    2.  **Smart Chat AI** (Controller):
        *   Interprets Natural Language ("I'm bored").
        *   Handles Image Uploads.
    3.  **Hybrid Engine** (Backend):
        *   **Collaborative**: SVD (User History).
        *   **Content-Based**: TF-IDF (Metadata).

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"So, how did we solve this? Presenting **'fmov'**, our solution.
We didn't want to build just a model in a Jupyter Notebook; we wanted a full Application. The solution rests on three main pillars.
**Pillar 1: The Experience (Frontend)**.
We built a custom UI using Streamlit, but we heavily modified it with CSS to move away from the standard 'Data Science Dashboard' look.
We implemented a **Dark Theme** similar to Netflix or Prime Video because that is what users expect in a media app.
We replaced the standard sidebar with a **Top Navigation Bar** and built a 'Card-based' layout where movies are tiles, not rows in a spreadsheet. This visual familiarity reduces the cognitive load on the user.
**Pillar 2: The Brain (Smart Chat AI)**.
This is the most novel part of our system. Instead of just a search bar, we implemented an 'Agent'.
This agent, which we'll discuss in the architecture slide, sits between the user and the database.
It is capable of **Reasoning**. If you say 'Show me movies,' it knows that's a SEARCH request.
If you say 'I want to watch something exciting,' it categorizes that as a RECOMMENDATION request and maps 'Exciting' to genres like 'Action' and 'Thriller'.
It effectively translates human language into database queries.
**Pillar 3: The Engine (Hybrid Backend)**.
We didn't rely on just one algorithm. We used a **Hybrid** approach.
We use **SVD (Singular Value Decomposition)** for Collaborative Filtering. This learns from the patterns of other users—if users who liked *Iron Man* also liked *Captain America*, the model learns that latent connection.
But we also use **TF-IDF (Term Frequency-Inverse Document Frequency)** for Content-Based Filtering. This analyzes the plot description.
If a movie is about 'Space' and 'Time Travel', TF-IDF finds other movies with those keywords.
By checking both, 'fmov' gives you recommendations that are both **Popular** (what others liked) and **Relevant** (what matches the content).
This 3-pillar structure ensures we solve the user interface problem, the intent understanding problem, and the recommendation accuracy problem simultaneously."

---

## Slide 4: System Architecture

### **Visual Content**
*   **Diagram**:
    *   **User Layer**: Web Browser -> Streamlit Cloud.
    *   **Controller Layer**: `SmartChatAI` Class (The Router).
    *   **Model Layer**:
        *   `BART-MNLI` (Intent Classification).
        *   `CLIP` (Image Processing).
        *   `SBERT` (Semantic Search).
        *   `SVD` (Rating Prediction).
    *   **Data Layer**: Kaggle CSV + Synthetic Reviews.

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"Let's look under the hood. This architecture diagram illustrates the data flow within fmov.
It follows a standard Model-View-Controller (MVC) pattern, adapted for AI applications.
**1. The User Layer (View)**:
The user interacts with the Streamlit frontend. Every click, chat message, or image upload triggers a session state update. We use 'Session State' to maintain history, so if you switch from Home to Chat, your conversation isn't lost.
**2. The SmartChatAI (Controller)**:
This is the central logic unit. When a user sends a message, it doesn't go straight to the database. It goes to the **SmartChatAI Router**.
This router uses a small **Zero-Shot Classification Model** (`facebook/bart-large-mnli`). It asks the model: 'Is this text a *Search query*, a *Recommendation request*, or *Chit-chat*?'
Based on the probability score, it routes the request.
**3. The Model Layer**:
If it's a **Visual Search** (Image Upload), we invoke **OpenAI CLIP**. CLIP converts the image into a 512-dimensional vector and finds text descriptions that match that vector.
If it's a **Semantic Search**, we use **SBERT (Sentence-BERT)**. It converts the user's sentence into a dense vector embedding and searches our movie description embeddings. This is far more powerful than simple keyword matching.
For the core recommendations, we use the **Hybrid Engine**.
The **SVD Model** creates a matrix of User vs. Item latent factors. It predicts the 'rating' a user would give to every movie they haven't seen.
The **Content Engine** calculates Cosine Similarity between the movie's metadata tags.
We combine these scores: `Final_Score = 0.6 * SVD_Score + 0.4 * Content_Score`.
**4. The Data Layer**:
We rely on the processed Netflix dataset (8,800 titles). Since this dataset lacks user ratings, we procedurally generated **Synthetic Ratings** for 500 users, clustering them into profiles like 'Horror Fan' or 'Comedy Fan' to accurately train the SVD model.
This layered architecture ensures modularity—we can swap out the BERT model for a Llama 3 model later without breaking the frontend."

---

## Slide 5: System Design (The "Smart Chat" Agent)

### **Visual Content**
*   **Workflow**:
    1.  User Input: "I want a scary movie about ghosts" OR [Uploads Image].
    2.  **Router**: Detects Intent = `RECOMMENDATION` (Conf: 0.92).
    3.  **Extractor**: Extracts Genre=`Horror`, Keywords=`Ghosts`.
    4.  **Retrieval**: Hybrid Engine fetches top 10 matches.
*   **Technologies**:
    *   `Transformers` (Hugging Face)
    *   `Lazy Loading` (@st.cache_resource)

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"I want to zoom in on the **Smart Chat Agent**, as it is the most complex component.
Designing this required solving the problem of 'Ambiguity'.
In a traditional system, you have dropdowns for Genre. But in Chat, a user can say anything.
We designed a pipeline that runs in real-time:
**Step 1: Ingestion & Lazy Loading**:
One challenge we faced was that loading heavy Transformer models (like BERT or CLIP) takes time and RAM.
We implemented a **Lazy Loading** design pattern. The application starts instantly. The heavy AI models are only loaded into memory the *first time* a user actually opens the Chat tab.
We use `@st.cache_resource` to keep these models in RAM so subsequent queries are instant (milliseconds).
**Step 2: Intent Classification**:
We feed the user's text into the BART-MNLI model with candidate labels: `['search', 'recommendation', 'greeting']`.
For the query 'I want a scary movie', the model assigns a high probability to `recommendation`.
For 'Find The Godfather', it assigns high probability to `search`.
This allows us to switch algorithms dynamically. Recommendations prioritize our SVD model (personalization), while Searches prioritize our TF-IDF model (accuracy).
**Step 3: Entity Extraction**:
Once we know it's a recommendation, we need filters.
We use a combination of **REGEX (Regular Expressions)** for speed and **NLP** for complexity.
For 'scary', our dictionary maps it to 'Horror' and 'Thriller'.
For 'ghosts', it becomes a semantic keyword.
**Step 4: Image Processing**:
If an image is uploaded, we don't treat it as a file. We pass it through **CLIP (Contrastive Language-Image Pre-Training)**.
CLIP 'reads' the image and outputs text probabilities. If you upload a picture of a space station, CLIP outputs 'Sci-Fi', 'Space', 'Future'.
We then feed these text tags back into our standard search engine. This effectively translates pixels into a database query.
This pipeline allows 'fmov' to handle inputs that are unstructured, ambiguous, or even non-textual."

---

## Slide 6: Demo Snapshots (Visuals)

### **Visual Content**
*   **Image 1**: The **Home Page**.
    *   Shows the "Featured Movie" banner (Hero Section) with a high-res background.
    *   Shows the "Trending Now" row.
*   **Image 2**: The **Chat Interface**.
    *   User text: "Show me something emotional."
    *   AI Reply: "I recommend *The Pursuit of Happyness* - it matches your 'emotional' request and has a high drama score."
*   **Image 3**: **Vision Search**.
    *   Shows an uploaded image of a starry sky.
    *   Shows results: *Interstellar*, *Gravity*, *Apollo 13*.

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"Let's look at the system in action.
**Snapshot 1: The Home Dashboard**.
This is the first screen the user sees. Notice the absence of clutter. We don't show a table of 8,000 rows.
We show a curated **Hero Section**. This is selected dynamically—our code checks for 'Recent & Highly Rated' movies and selects one to feature.
Below it, you see the 'Feature Cards' for Navigation. The design language here uses Shadows, Hover Effects, and Gradients (Deep Red to Black) to mimic the premium feel of Netflix.
**Snapshot 2: The Conversational AI**.
Here you see the chat in action. The user asks, 'Show me something emotional.'
Notice the AI's response. It doesn't just dump a list. It generates a natural language explanation: *'I recommend... because it matches your request.'*
This **Explainability** is crucial. Users trust AI recommendations more when they understand *why* a suggestion was made.
Under the hood, the system mapped 'emotional' to the 'Drama' genre and applied a 'Sad/Moving' mood filter.
**Snapshot 3: Visual Search**.
In this example, we uploaded a generic image of a starry night sky.
The system didn't look for the filename 'stars.jpg'.
The CLIP model analyzed the pixel data, identified concepts like 'Space', 'Astronomy', and 'Dark'.
It then retrieved *Interstellar* and *Gravity*.
This is a powerful feature for when users have a visual memory of a movie ('I remember a scene with a red robot') but don't know the name.
These snapshots validate that all three pillars of our solution—UI, Chat, and Vision—are functioning cohesively."

---

## Slide 7: Results & Performance

### **Visual Content**
*   **Quantitative Metrics**:
    *   **RMSE**: 1.38 (Root Mean Square Error).
    *   **MAE**: 1.12 (Mean Absolute Error).
*   **Performance Metrics**:
    *   **Startup Time**: Reduced from 45s -> 3s (Lazy Loading).
    *   **Inference Time**: ~200ms for Text Search.
*   **Qualitative**:
    *   Hybrid Model > content-based Model (Subjective quality).
    *   "Surprise Me" feature increases catalog coverage.

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"Beyond the visuals, we rigorously evaluated the system's performance.
**Accuracy Metrics**:
We used **RMSE (Root Mean Square Error)** to evaluate the Collaborative Filtering component.
We achieved an RMSE of **1.38** on a 5-point scale. This means our predicted rating is usually within 1.3 stars of the user's actual preference.
While this sounds modest, in the context of sparse datasets (where 99% of the matrix is empty), this is a solid baseline. It confirms the SVD model successfully learned the latent clusters we generated in our synthetic data.
**System Performance (Latency)**:
One major hurdle in AI engineering is speed.
Initially, our app took 45 seconds to load because it was downloading the 1.5GB BERT models on startup.
This was unacceptable. We optimized this by deferring the load.
Now, the app loads in **3 seconds**. The models load in the background only when needed.
The inference time for a text search is roughly **200 milliseconds**, which is perceived as 'instant' by the user.
**Qualitative Quality**:
We performed A/B testing (simulated) between the Content-Based model alone and the Hybrid model.
The Content-Based model suffered from the 'Harry Potter Problem'—if you watched Harry Potter, it only recommended fantasy movies.
The Hybrid model, however, introduced **Serendipity**. It noticed that users who liked *Harry Potter* also tended to like *Disney Animation*, even though the genres (Fantasy vs. Animation) are different.
This ability to make non-obvious connections is the key value add of our Hybrid approach.
Finally, our 'Surprise Me' button ensures that we don't trap users in a 'Filter Bubble', artificially exposing them to high-quality content they might otherwise miss."

---

## Slide 8: LLM Evaluation (Agentic AI)

### **Visual Content**
*   **Goal**: Validate "Router" Logic (BART-MNLI).
*   **Test Cases**:
    *   Case A: "Find *Inception*" -> `SEARCH` ✅
    *   Case B: "I'm sad, cheer me up" -> `RECOMMEND` ✅
    *   Case C: "Movies with Tom Cruise" -> `SEARCH` + Entity: `Actor` ✅
*   **Confidence Scores**:
    *   Clear intents have ~0.9+ confidence.
    *   Ambiguous intents fall back to Keyword Search.

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"Evaluating Generative AI and Classifiers is harder than evaluating simple regression models. You can't just calculate RMSE for a chat bot.
To evaluate our **Smart Chat Agent**, we used a 'Test Case' approach to verify the Router's logic.
**Case A: Explicit Search**.
Query: 'Find Inception'.
The model correctly identified this as a `SEARCH` intent with 95% confidence. It didn't try to recommend 'movies similar to Inception'; it fetched the exact title. This prevents user frustration when they know exactly what they want.
**Case B: Abstract Mood**.
Query: 'I'm sad, cheer me up'.
This is the hardest case for traditional logic. There is no movie titled 'Cheer me up'.
Our Zero-Shot model successfully classified this as a `RECOMMENDATION` request.
Furthermore, our keyword mapping logic associated 'Cheer me up' with the 'Comedy' and 'Family' genres.
The result was a list of feel-good movies. This proves the **semantic understanding** of the Agent.
**Case C: Specific Entity**.
Query: 'Movies with Tom Cruise'.
The Agent correctly identified 'Tom Cruise' as a named entity (Actor).
Instead of searching for 'Tom Cruise' in the *Title* column (which would yield zero results), it switched the search target to the *Cast* column.
This dynamic column switching is a feature that basic search bars lack.
**Conclusion**:
The usage of the BART-MNLI model as a 'Zero-Shot Classifier' allows us to route queries with near-human accuracy without having to train a custom model on thousands of labeled chat logs. It's a highly efficient implementation of Agentic AI."

---

## Slide 9: Key Learnings

### **Visual Content**
*   **1. Hybrid > Single Model**: Combining algorithms covers blind spots (Cold Start vs. Filter Bubbles).
*   **2. Semantic Search is Critical**: Keywords are limiting. Vector embeddings unlock "Concept Search".
*   **3. UX Drives Engagement**: A good model in a bad UI is useless. The "Netflix" aesthetic builds trust.
*   **4. Optimization**: Lazy Loading is non-negotiable for Python web apps using Transformers.

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"Building 'fmov' taught us several critical lessons about AI Engineering.
**Learning 1: The Power of Hybrid Systems**.
We learned that no single algorithm is perfect. SVD is great but fails for new users. TF-IDF is stable but boring.
Combining them (Ensemble Learning) creates a system that is robust. It covers the weaknesses of each individual approach.
**Learning 2: Semantics over Syntax**.
The shift from 'Keyword Search' to 'semantic Search' (Embeddings) was eye-opening.
Previously, searching for 'Aliens' wouldn't find *War of the Worlds* if the description only said 'Extraterrestrial'.
With SBERT embeddings, the system understands that 'Alien' and 'Extraterrestrial' are close in vector space. This fundamentally changes the quality of search results.
**Learning 3: The Importance of UX/UI**.
We spent a significant amount of time on CSS and Layout. Why?
Because Recommendation is a distinctively *visual* experience. Users judge a movie by its cover.
Presenting data in a CSV format vs. a Card format changes user perception of the system's intelligence. A polished UI builds trust in the AI's recommendations.
**Learning 4: Production Engineering**.
Getting this to run smoothly in a browser required engineering optimization.
We learned about Python's memory management, the importance of caching, and the 'Lazy Loading' pattern.
We realized that loading a 1GB model just to render the 'Home Page' is wasteful. Deferring that load until the user clicks 'Chat' makes the app feel significantly snappier.
These learnings move us from being 'Model Builders' to 'System Architects'."

---

## Slide 10: Conclusion & Future Scope

### **Visual Content**
*   **Conclusion**:
    *   "fmov" bridges the gap between Choice Paralysis and Discovery.
    *   Demonstrates providing "Concierge" experiences using Agentic AI.
*   **Future Scope**:
    *   ☁️ **Cloud Deployment** (Docker/AWS).
    *   🔄 **Real-Time Learning** (Online SVD Updates).
    *   🗣️ **Voice Interface** (Whisper AI Integration).
    *   📹 **Video Analysis** (Analyzing Trailers).

### **Detailed Speaker Notes & Explanation (approx. 50 lines)**
"To conclude, **'fmov'** successfully demonstrates that we can build a recommendation system that is not just accurate, but **Intelligent**.
We tackled the 'Paradox of Choice' by building a system that listens, sees, and understands.
We moved beyond the constraints of SQL-like queries into the world of Semantic and Visual search.
We delivered a user experience that rivals commercial platforms like Netflix, proving that advanced AI can be accessible and intuitive.
**Where do we go from here?**
This project lays the foundation for several exciting expansions:
1.  **Cloud Deployment**: Currently, this runs locally. The next step is to Dockerize the application and deploy it to a scalable architecture like **AWS ECS** or **Google Cloud Run**, allowing it to handle thousands of concurrent users.
2.  **Real-Time Learning**: Right now, our SVD model is static. We want to implement an **Online Learning** loop. If a user rates a movie *now*, the matrix should update *instantly*, improving their very next recommendation.
3.  **Multi-Modal Expansion**: We want to go beyond images. We could use models to analyze the actual *video trailers* to extract mood and pacing, giving us even deeper metadata than just the plot description.
4.  **Voice Interaction**: Finally, integrating **Whisper AI** would allow users to just speak to the app—'Hey fmov, find me something funny'—making the 'Concierge' experience complete.
'fmov' is just the beginning of how Agentic AI will reshape content discovery.
Thank you for listening. I am open to any questions."
