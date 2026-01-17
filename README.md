# 🎬 fmov: Intelligent Content Recommendation System & AI Chat
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Transformers-yellow.svg)](https://huggingface.co)

> A Netflix-style media platform featuring a **Hybrid Recommendation Engine** and an **Agentic AI Chatbot** capable of Mood Analysis, Computer Vision, and Semantic Search.

## 🎯 Overview
**fmov** (formerly FlixMood) is a next-generation content discovery platform. It solves the problem of "choice paralysis" by combining traditional collaborative filtering with advanced AI.

### ✨ Key Features
| Feature | Tech Stack | Description |
|---------|------------|-------------|
| **🤖 Smart Chat AI** | `BERT`, `LLM` | Ask "I want a scary movie about ghosts" or "I'm bored". The AI extracts intent and filters content. |
| **👁️ Vision Analysis** | `CLIP` (OpenAI) | **Upload an image** (e.g., a movie poster or scene) and the AI will find similar content. |
| **🎭 Mood & Context** | `Hybrid Engine` | Filter by Mood (Happy, Intense) or Context (Date Night, Family Time). |
| **� Netflix UI** | `Streamlit` + `CSS` | A modern, dark-themed "fmov" interface with a responsive card-based dashboard. |
| **🔍 Semantic Search** | `SBERT` | Search by meaning, not just keywords (e.g., "movies about space travel"). |
| **📓 Full Notebook** | `Jupyter` | Includes `FlixMood_Project.ipynb` for a complete code walkthrough. |

---

## 🏗️ System Architecture

1.  **Presentation Layer**: Streamlit (Python web framework) with custom "fmov" CSS.
2.  **Controller Layer**: `SmartChatAI` class (Router) determining if user input is a Search, Recommendation Request, or General Chat.
3.  **Model Layer**:
    *   **Vision**: `openai/clip-vit-base-patch32`
    *   **Reasoning**: `facebook/bart-large-mnli` (Zero-Shot)
    *   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
    *   **Recommender**: SVD (Matrix Factorization) + TF-IDF (Content-Based)

---

## 🚀 Installation & Setup

### Prerequisites
*   Python 3.8+
*   Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fmov-recommendation-system.git
cd fmov-recommendation-system
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: This will install Streamlit, PyTorch, Transformers, and other ML libraries.*

### 4. Run the Application
```bash
streamlit run app.py
```
*   The first run will download necessary AI models (~2-3 GB). Please be patient.
*   Access the app at `http://localhost:8501`.

---

## 📘 Jupyter Notebook Walkthrough
For a step-by-step explanation of the code, data analysis, and model training, open the included notebook:
**`FlixMood_Project.ipynb`**

You can run this in VS Code or Jupyter Lab.

---

## � Project Structure
```text
fmov/
├── app.py                 # Main Application Entry Point
├── create_notebook.py     # Script to generate the project notebook
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── src/
│   ├── ui_components.py   # UI/UX (Navbar, Cards, CSS)
│   ├── smart_chat_ai.py   # AI Logic (CLIP, BART, SBERT)
│   ├── data_loader.py     # Dataset Management
│   ├── collaborative.py   # SVD Model
│   ├── content_based.py   # TF-IDF Model
│   ├── hybrid.py          # Hybrid Logic
│   └── evaluation.py      # Metrics
└── data/                  # Cached Data
```

## � Dataset
Using the **Netflix Movies and TV Shows** dataset from Kaggle (`shivamb/netflix-shows`).
*   **Size**: ~8,800 titles.
*   **Enhancement**: We generate **Synthetic Ratings** (`synthetic_ratings.csv`) to simulate realistic user behavior (1-5 stars) for training the Collaborative Filtering model.

## 📝 License
MIT License. Built for educational purposes.
