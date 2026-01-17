"""
Smart Chat AI Controller
========================
Interprets natural language input, detects intent (SEARCH vs RECOMMEND),
and constructs structured queries for the content recommendation engine.

This is a CONTROLLER, not a Model. It does not perform recommendation logic itself.
"""

from typing import Dict, List, Optional, Any
import re
import streamlit as st

# Lazy import for Hugging Face to avoid startup lag if libraries missing
try:
    import torch
    from sentence_transformers import SentenceTransformer, util
    from transformers import pipeline, CLIPProcessor, CLIPModel
    from PIL import Image
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

@st.cache_resource(show_spinner="Loading Semantic Brain...")
def get_semantic_model():
    """Load SBERT model lazily."""
    try:
        if not HF_AVAILABLE: return None
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Failed to load Semantic Model: {e}")
        return None

@st.cache_resource(show_spinner="Loading Vision Brain...")
def get_vision_models():
    """Load CLIP processor and model lazily."""
    try:
        if not HF_AVAILABLE: return None, None
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        return processor, model
    except Exception as e:
        print(f"Failed to load Vision Model: {e}")
        return None, None

@st.cache_resource(show_spinner="Loading Intent Brain...")
def get_intent_pipeline():
    """Load Zero-Shot classifier lazily."""
    try:
        if not HF_AVAILABLE: return None
        # Using a smaller distilled model for speed
        return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
    except Exception as e:
        print(f"Failed to load Intent Model: {e}")
        return None

class SmartChatAI:
    def __init__(self):
        """
        Initialize rule sets. Models are loaded on demand via cached functions.
        """
        # --- Intent Patterns (Rule-based Phase 1) ---
        self.intent_patterns = {
            "SEARCH": [
                r"\bfind\b", r"\bsearch\b", r"\bshow me\b", r"\bmovies by\b", 
                r"\blist\b", r"\bcast\b", r"\bdirector\b", r"\bwho is\b", r"\bwhere can i watch\b",
                r"\blookup\b"
            ],
            "RECOMMEND": [
                r"\bsuggest\b", r"\brecommend\b", r"\bi like\b", r"\bfeel like\b", 
                r"\bwant something\b", r"\bwhat should i watch\b", r"\bgive me\b",
                r"\bbored\b", r"\bmood\b", r"\bi want to watch\b", r"\blooking for\b",
                r"\bany good\b"
            ]
        }
        
        # Known genres for entity extraction
        self.known_genres = [
            "Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", 
            "Documentary", "Thriller", "Animation", "Family", "Crime", 
            "Adventure", "Fantasy", "Mystery", "History"
        ]
        
        # Stopwords to clean for keyword extraction (basic list)
        self.stopwords = {
            "a", "an", "the", "in", "on", "at", "for", "to", "of", "with", 
            "and", "or", "is", "are", "was", "were", "be", "been", "being",
            "movies", "movie", "show", "shows", "tv", "series", "film", "films",
            "something", "like", "about", "find", "search", "recommend", "suggest",
            "want", "watch", "looking"
        }

    # No explicit load method needed on init, we call cached functions

    def process_input(self, user_text: str, image_file=None) -> Dict[str, Any]:
        """
        Main entry point. Processes input and determines execution plan.
        """
        response = {
            "mode": "NOT_FOUND",
            "query": {
                "text": user_text,
                "keywords": [],
                "genres": [],
                "actor": "",
                "format": ""
            },
            "explanation": "I couldn't quite understand what you're looking for."
        }
        
        if not user_text and not image_file:
            return response

        # 1. Detect Intent
        intent = self._detect_intent(user_text, image_file)
        response["mode"] = intent
        
        # 2. Extract Entities
        entities = self._extract_entities(user_text)
        response["query"].update(entities)
        
        # 3. SEMANTIC SEARCH (Lazy Load)
        if HF_AVAILABLE:
            semantic_model = get_semantic_model()
            if semantic_model:
                query_embedding = semantic_model.encode(user_text, convert_to_tensor=True)
                response["query"]["embedding"] = query_embedding.tolist()
                response["query"]["hf_enabled"] = True
            
        # 4. VISION UPGRADE (Lazy Load)
        if HF_AVAILABLE and image_file:
            image_analysis = self._analyze_image(image_file)
            if image_analysis:
                response["query"].update(image_analysis)
                response["explanation"] += f" I analyzed the image and found: {', '.join(image_analysis.get('keywords', [])[:3])}."
        
        # 5. INTENT REASONING (Lazy Load)
        # Only use LLM if simple rules failed or result is Vague/Not Found
        if response["mode"] == "NOT_FOUND" and HF_AVAILABLE:
            intent_pipeline = get_intent_pipeline()
            if intent_pipeline:
                llm_intent = self._reason_intent(user_text, intent_pipeline)
                if llm_intent != "NOT_FOUND":
                    response["mode"] = llm_intent
                    response["explanation"] = "I used deep reasoning to understand your request."
        
        # 6. Construct Explanation & Finalize
        response = self._construct_response(response)
        
        return response

    def _analyze_image(self, image_file) -> Dict[str, Any]:
        """Analyze image using CLIP to find visual keywords."""
        try:
            processor, model = get_vision_models()
            if not model or not processor: return {}
            
            image = Image.open(image_file)
            
            # Zero-shot classification with CLIP
            visual_concepts = ["space", "action", "romance", "nature", "horror", "city", "future"] 
            labels = self.known_genres + visual_concepts
            
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            
            # Get top labels
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # Get top 3 indices
            top_indices = probs.topk(3).indices[0].tolist()
            top_labels = [labels[i] for i in top_indices]
            
            return {
                "keywords": top_labels,
                "genres": [lbl for lbl in top_labels if lbl in self.known_genres]
            }
        except Exception as e:
            print(f"Vision analysis failed: {e}")
            return {}

    def _reason_intent(self, text: str, pipeline_obj=None) -> str:
        """Use LLM (or Zero-Shot) to deduce intent from vague text."""
        try:
            if not pipeline_obj: return "NOT_FOUND"
            
            candidate_labels = ["search for a specific movie", "request for recommendations", "ask about an actor"]
            result = pipeline_obj(text, candidate_labels)
            
            top_label = result['labels'][0]
            score = result['scores'][0]
            
            # Lowered threshold for better recall with the distilled model
            if score > 0.4: 
                if "search" in top_label or "actor" in top_label:
                    return "SEARCH"
                elif "recommend" in top_label:
                    return "RECOMMEND"
            
            return "NOT_FOUND"
        except Exception as e:
            print(f"LLM reasoning failed: {e}")
            return "NOT_FOUND"

    def _detect_intent(self, text: str, image_file=None) -> str:
        """
        Determine if user wants SEARCH or RECOMMEND.
        """
        if image_file:
            return "SEARCH"  # Implicitly Phase 1 logic: Image -> Find this
            
        text_lower = text.lower()
        
        # Check explicit triggers
        for pattern in self.intent_patterns["SEARCH"]:
            if re.search(pattern, text_lower):
                return "SEARCH"
                
        for pattern in self.intent_patterns["RECOMMEND"]:
            if re.search(pattern, text_lower):
                return "RECOMMEND"
        
        # Implicit Detection
        
        # A. Capitalized phrases (likely titles/names) -> SEARCH
        # Example: "Mission Impossible", "Tom Cruise"
        if any(word[0].isupper() for word in text.split() if len(word) > 1 and word.lower() not in self.stopwords):
             # Ensure it's not just a mood started with capital (e.g. "Happy movies")
             # Heuristic: If meaningful capital letters exist, lean towards Search
             pass 

        # B. Mood words -> RECOMMEND
        mood_words = ["funny", "scary", "sad", "intense", "thrilling", "relaxing", "uplifting"]
        if any(w in text_lower for w in mood_words):
            return "RECOMMEND"
            
        # C. Short queries that look like titles -> SEARCH
        # "Inception", "The Matrix"
        if len(text.split()) < 5:
            return "SEARCH"
            
        return "NOT_FOUND"

    def _extract_entities(self, text: str) -> Dict:
        """
        Extract relevant signals from text.
        """
        text_lower = text.lower()
        words = text.split()
        
        extracted = {
            "keywords": [],
            "genres": [],
            "actor": "", # Simplified: assume capitalized words could be actors in Phase 1
            "format": ""
        }
        
        # Extract Genres
        for genre in self.known_genres:
            if genre.lower() in text_lower:
                extracted["genres"].append(genre)
        
        # Extract Keywords (Simple stopword removal)
        clean_words = [w for w in text_lower.split() if w not in self.stopwords]
        # Remove genre words from keywords to avoid duplication
        genre_lower = [g.lower() for g in self.known_genres]
        clean_words = [w for w in clean_words if w not in genre_lower]
        
        extracted["keywords"] = clean_words
        
        # Extract potential Actor/Title (Capitalized phrases)
        # This is a basic heuristic for Phase 1
        capitalized = [w for w in words if w and w[0].isupper()]
        # Filter out common capitalized stopwords if any (at start of sentence)
        if capitalized:
            # Join consecutive capitalized words? "Tom Cruise"
            # For now just store as raw keywords or text.
            # We can tentatively put them in actor if "by" precedes it
            pass
            
        if "movie" in text_lower or "film" in text_lower:
            extracted["format"] = "Movie"
        elif "tv" in text_lower or "series" in text_lower or "show" in text_lower:
            extracted["format"] = "TV Show"
            
        return extracted

    def _construct_response(self, response: Dict) -> Dict:
        """
        Finalize query and add human-readable explanation.
        """
        mode = response["mode"]
        q = response["query"]
        
        # Build Explanation
        if mode == "SEARCH":
            if q["genres"]:
                response["explanation"] = f"I'm searching for {', '.join(q['genres'])} matches in the catalog."
            elif q["keywords"]:
                keywords_str = ", ".join(q["keywords"][:3])
                response["explanation"] = f"I'm searching for titles matching specific terms: '{keywords_str}'."
            else:
                response["explanation"] = "I'm looking up that specific title in the library."
                
        elif mode == "RECOMMEND":
            reasons = []
            if q["genres"]:
                reasons.append(f"{', '.join(q['genres'])}")
            if q["keywords"]:
                reasons.append("your keywords")
            
            if reasons:
                response["explanation"] = f"I'm recommending content based on {' and '.join(reasons)}."
            else:
                response["explanation"] = "I'm checking our top recommendations for you."
                
        elif mode == "NOT_FOUND":
            response["explanation"] = "I wasn't sure if you wanted to search or get recommendations. Try typing a title or a mood!"
            
        return response
