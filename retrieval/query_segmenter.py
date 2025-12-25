# # query_segmenter.py
# import re
# import time
# import spacy
# from langdetect import detect
# from transformers import pipeline

# class QuerySegmenter:
#     """
#     Analyzes a user query to classify its intent and extract ucd seful slots.
#     This class acts as a router to determine the best downstream pipeline.
#     """

#     def __init__(self, confidence_threshold=0.6, use_gpu=False):
#         print("Initializing QuerySegmenter...")
        
#         # --- Configuration ---
#         self.confidence_threshold = confidence_threshold
#         self.device = 0 if use_gpu else -1

#         # --- Pipeline Mapping ---
#         self.pipeline_map = {
#             "factual": "hybrid_retrieval_concise",
#             "definition": "retrieval_template",
#             "explain": "multi_chunk_retrieval_generator",
#             "compare": "dual_retrieval_comparison",
#             "math": "math_solver",
#             "diagram": "visual_q_and_a",
#             "procedural": "stepwise_retrieval_generator",
#             "summary": "full_chapter_summarizer",
#             "follow_up": "contextual_retrieval",
#             "multi_hop": "knowledge_graph_retrieval",
#             "translation": "translation_model",
#             "creative": "long_form_generator",
#             "faq_lookup": "faq_retrieval"
#         }
        
#         # Define candidate labels for zero-shot classification
#         self.candidate_labels = list(self.pipeline_map.keys())

#         # --- Load Models (lazy loading can be implemented if startup is slow) ---
#         print("Loading spaCy model for NER...")
#         self.nlp = spacy.load("en_core_web_sm")

#         print("Loading zero-shot classification pipeline...")
#         # Using a distilled, faster model suitable for this task
#         self.zero_shot_classifier = pipeline(
#             "zero-shot-classification",
#             model="valhalla/distilbart-mnli-12-3",
#             device=self.device
#         )
#         print("✅ QuerySegmenter initialized successfully.")


#     def _detect_language_and_normalize(self, query: str) -> str:
#         """Detects language and normalizes query. Placeholder for transliteration."""
#         try:
#             lang = detect(query)
#             # In a real system, you might transliterate Hinglish to Hindi or English here
#             if lang != 'en':
#                 print(f"⚠️ Language detected: {lang}. Only English is fully supported.")
#         except Exception:
#             print("Language detection failed. Assuming English.")
        
#         return query.lower().strip()

#     def _extract_slots(self, query: str) -> dict:
#         """Extracts entities, numbers, and other signals from the query."""
#         doc = self.nlp(query)
#         slots = {
#             "entities": [ent.text for ent in doc.ents],
#             "numbers": [tok.text for tok in doc if tok.like_num],
#             "class_hint": re.findall(r'class\s*(\d+)', query),
#             "chapter_hint": re.findall(r'chapter\s*(\d+)', query),
#             "q_word": next((tok.text for tok in doc if tok.tag_ in ['WDT', 'WP', 'WP$', 'WRB']), None),
#             "has_math_symbols": bool(re.search(r'[\d\+\-\*\/=∑√%]', query)),
#             "is_image_request": bool(re.search(r'\b(figure|image|diagram|draw|picture|fig\.)\b', query))
#         }
#         return slots

#     def _apply_rules(self, query: str, slots: dict, chat_history: list = None) -> dict | None:
#         """Applies high-precision rules to identify clear-cut segments."""
#         # Rule for Math
#         if slots["has_math_symbols"] or re.search(r'\b(solve|calculate|area|perimeter|what is)\b.*\d', query):
#             return {"segment": "math", "confidence": 1.0}

#         # Rule for Diagram/Image requests
#         if slots["is_image_request"]:
#             return {"segment": "diagram", "confidence": 1.0}

#         # Rule for Summary
#         if query.startswith("summarize") or "summary of" in query:
#             return {"segment": "summary", "confidence": 1.0}

#         # Rule for Definition
#         if query.startswith("define") or query.startswith("what is a") or query.startswith("what is an"):
#             return {"segment": "definition", "confidence": 1.0}
            
#         # Rule for Comparison
#         if "compare" in query or "difference between" in query:
#              return {"segment": "compare", "confidence": 1.0}

#         # Rule for Follow-up
#         # A short query with a pronoun is a strong indicator if chat history exists.
#         first_token = query.split()[0] if query else ""
#         if chat_history and len(query.split()) < 5 and first_token in ["what", "who", "where", "how", "why", "he", "she", "it", "they"]:
#             return {"segment": "follow_up", "confidence": 0.95}

#         return None

#     def _zero_shot_classify(self, query: str) -> dict:
#         """Uses a zero-shot model to classify the query into a segment."""
#         result = self.zero_shot_classifier(query, self.candidate_labels, multi_label=False)
#         return {
#             "segment": result['labels'][0],
#             "confidence": result['scores'][0]
#         }

#     def _finetuned_classify(self, query: str) -> dict:
#         """Placeholder for a future fine-tuned classification model."""
#         print("LOG: Fine-tuned model not implemented. Skipping.")
#         # When implemented, this would take precedence over zero-shot.
#         # Example:
#         # logits = self.finetuned_model.predict(query)
#         # probabilities = softmax(logits)
#         # top_class_idx = argmax(probabilities)
#         # return {
#         #     "segment": self.class_labels[top_class_idx],
#         #     "confidence": probabilities[top_class_idx]
#         # }
#         return None

#     def segment(self, query: str, chat_history: list = None) -> dict:
#         """
#         Main method to process a query and return a structured routing object.

#         Args:
#             query (str): The user's input query.
#             chat_history (list, optional): A list of previous conversation turns. Defaults to None.

#         Returns:
#             dict: A structured object with segment, confidence, slots, and suggested pipeline.
#         """
#         start_time = time.time()
        
#         # 1. Normalize and Extract Slots
#         normalized_query = self._detect_language_and_normalize(query)
#         slots = self._extract_slots(normalized_query)
        
#         # 2. Apply Rule-based Heuristics
#         result = self._apply_rules(normalized_query, slots, chat_history)

#         # 3. If no rule matches, use ML models
#         if not result:
#             # Try fine-tuned model first (if implemented)
#             result = self._finetuned_classify(normalized_query)
            
#             # Fallback to zero-shot model
#             if not result:
#                 result = self._zero_shot_classify(normalized_query)

#         # 4. Confidence Check and Fallback
#         if result['confidence'] < self.confidence_threshold:
#             print(f"LOG: Low confidence ({result['confidence']:.2f}) for '{result['segment']}'. Falling back to 'factual'.")
#             result['segment'] = 'factual'
#             # We can keep the original low confidence score for logging/analysis
        
#         # 5. Build final structured output
#         final_decision = {
#             "original_query": query,
#             "normalized_query": normalized_query,
#             "segment": result['segment'],
#             "confidence": result['confidence'],
#             "slots": slots,
#             "suggested_pipeline": self.pipeline_map.get(result['segment'], "default_rag")
#         }

#         end_time = time.time()
#         print(f"LOG: Segmentation complete in {end_time - start_time:.4f}s. Decision: {final_decision['segment']} (Conf: {final_decision['confidence']:.2f})")
        
#         return final_decision

# # --- Example Usage ---
# if __name__ == "__main__":
#     # Initialize the segmenter (can be a singleton in a real app)
#     segmenter = QuerySegmenter()
    
#     # --- Test Cases ---
#     test_queries = [
#         "what is the area of a circle with radius 5cm?",
#         "summarize chapter 2 of the beehive book",
#         "define mitosis",
#         "explain the process of photosynthesis",
#         "what is the difference between prokaryotic and eukaryotic cells?",
#         "who was the first prime minister of India?",
#         "what about his main achievements?", # Needs history to be a follow_up
#         "label the parts in figure 4.1"
#     ]
    
#     print("\n--- Running Test Queries ---")
    
#     # Mock chat history for the "follow_up" example
#     history = [{"role": "user", "content": "who was jawaharlal nehru?"}]

#     for q in test_queries:
#         print(f"\n--- Query: '{q}' ---")
#         if "what about his" in q:
#             # Pass history for this specific query
#             output = segmenter.segment(q, chat_history=history)
#         else:
#             output = segmenter.segment(q)
        
#         # Pretty print the output
#         import json
#         print(json.dumps(output, indent=2))




# retrieval/query_segmenter.py
import re
from typing import Optional, Dict, Any, List

# Prefer lightweight heuristic classification during unit tests and CI so we
# avoid importing heavy transformer models at module import time.
ZERO_SHOT_AVAILABLE = False

# --- Config / labels ---
SEGMENTS = [
    "factual", "definition", "explain", "compare", "math",
    "diagram", "procedural", "summary", "follow_up", "multi_hop",
    "translation", "creative", "faq_lookup"
]

# Initialize zero-shot classifier if available (optional)
if ZERO_SHOT_AVAILABLE:
    try:
        from transformers import pipeline
        zsc = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    except Exception:
        ZERO_SHOT_AVAILABLE = False

# --- Rules / regex ---
MATH_PATTERNS = re.compile(r"(\d+|\d+\.\d+|solve|calculate|integral|differentiate|frac|sum|area|perimeter|x=|y=|\+|\-|\*|\/|=)")
DIAGRAM_KEYWORDS = re.compile(r"\b(figure|diagram|image|graph|map|chart|label|illustration|draw)\b", re.IGNORECASE)
COMPARE_KEYWORDS = re.compile(r"\b(compare|difference|contrast|how is .* different|vs\.|versus)\b", re.IGNORECASE)
SUMMARY_KEYWORDS = re.compile(r"\b(summar(y|ise|ize)|give a summary|in brief|summarise|briefly)\b", re.IGNORECASE)
TRANSLATION_KEYWORDS = re.compile(r"\b(translate|transliteration|convert to hindi|convert to english)\b", re.IGNORECASE)
FOLLOWUP_PRONOUNS = re.compile(r"\b(he|she|they|it|its|this|that|these|those|him|her)\b", re.IGNORECASE)

# Simple entity extraction (very lightweight)
ENTITY_PATTERN = re.compile(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b')


def get_query_segment(query: str) -> str:
    """
    Classify the query into one of the predefined segments.
    Uses simple keyword rules for now, but can be upgraded to ML classifier.
    """
    q = query.lower().strip()

    # Grammar correction queries
    if "correct this" in q or "grammar" in q or "fix sentence" in q:
        return "grammar"

    # Summarization queries
    if "summarize" in q or "summary" in q or "short note" in q:
        return "summarization"

    # Explanation queries
    if q.startswith("why") or q.startswith("how") or "explain" in q:
        return "explanation"

    # FAQ queries (simple keyword match for now)
    if "byjus" in q or "ncert" in q or "faq" in q:
        return "faq"

    # Image/diagram queries
    if "figure" in q or "diagram" in q or "map" in q or "image" in q:
        # pipeline expects 'image_based' or 'diagram' depending on caller; keep 'image_based'
        return "image_based"

    # Default: factual question answering
    # Historically the pipeline expects the label 'factual' in other places.
    # Return the canonical label 'factual' for consistency.
    return "factual"

def preprocess_query(text: str) -> str:
    return text.strip()

def extract_slots(query: str, chat_history: Optional[List[str]] = None) -> Dict[str, Any]:
    slots = {}
    # entities (capitalized multi-word sequences)
    # entities: capture capitalized named entities and also simple noun phrases
    ents = ENTITY_PATTERN.findall(query)
    # include nouns following phrases like 'of a/an/the X'
    simple_nouns = re.findall(r"of (?:a|an|the) ([a-zA-Z0-9_-]+)", query.lower())
    slots["entities"] = ents + simple_nouns
    # numbers found in the query
    slots["numbers"] = re.findall(r"\d+", query)
    # hint for class (e.g., 'class 9')
    m = re.search(r'\bclass\s*(\d{1,2})\b', query.lower())
    slots["class_hint"] = [m.group(1)] if m else []
    # question word (who/what/when/where/why/how/which)
    qw = re.search(r'\b(who|what|where|when|why|how|which)\b', query.lower())
    slots["q_word"] = qw.group(1) if qw else None
    # all W-words found (used by some fallback heuristics)
    slots["w_words"] = re.findall(r'\b(who|what|where|when|why|how|which)\b', query.lower())
    # whether the query contains math-like tokens or operators
    slots["has_math_symbols"] = bool(MATH_PATTERNS.search(query))
    return slots

def rule_based_segment(query: str, chat_history: Optional[List[str]] = None) -> Optional[str]:
    # Highest priority: follow-up if there's chat history
    if chat_history and FOLLOWUP_PRONOUNS.search(query) and len(query.split()) <= 6:
        return "follow_up"

    # Summary
    if SUMMARY_KEYWORDS.search(query):
        return "summary"

    # Comparison
    if COMPARE_KEYWORDS.search(query):
        return "compare"

    # Diagram/Image requests (prefer over explanation if image keywords present)
    if DIAGRAM_KEYWORDS.search(query):
        return "diagram"

    # Translation
    if TRANSLATION_KEYWORDS.search(query):
        return "translation"

    # Math detection should come before loose 'what is' definition detection
    if MATH_PATTERNS.search(query) and re.search(r"\bsolve\b|\bcalculate\b|=|\+|\-|\*|/|integral|differentiate|area|perimeter|fraction|%|\\/", query.lower()):
        return "math"

    # Procedural patterns (how-to steps, experiments)
    if re.search(r"\b(steps to|how do i|how to|procedure|list the steps|experiment|perform the experiment)\b", query.lower()):
        return "procedural"

    # Definition queries: 'define X' or 'what is a/an X' or 'what is X' but avoid 'what is it/this/that'
    # Definition only for 'define X' or 'what is a/an X'
    if re.match(r"^\s*(define\b|what is (?:a|an)\b)", query.lower()):
        return "definition"

    # Explanation (why/how/explain) as a fallback
    if query.lower().startswith("why") or query.lower().startswith("how") or "explain" in query.lower():
        return "explain"

    return None

def zero_shot_segment(query: str, candidate_labels=SEGMENTS):
    if ZERO_SHOT_AVAILABLE:
        out = zsc(query, candidate_labels)
        top_label = out['labels'][0]
        score = out['scores'][0]
        return top_label, float(score)

    # Lightweight heuristic fallback when transformers are not available.
    q = query.lower()
    # Creative (essay/poem)
    if any(w in q for w in ["essay", "poem", "compose", "write an essay", "write a"]):
        return "creative", 0.9
    # Translation
    if TRANSLATION_KEYWORDS.search(q):
        return "translation", 0.9
    # Procedural
    if any(w in q for w in ["steps to", "how do i", "procedure", "list the steps", "experiment"]):
        return "procedural", 0.85
    # Explain
    if q.startswith("why") or q.startswith("how") or "explain" in q or q.startswith("tell me about") or q.startswith("tell me more about"):
        return "explain", 0.8
    # Multi-hop heuristic
    if "which" in q and ("who" in q or "later" in q or "later won" in q):
        return "multi_hop", 0.75
    # Factual by default for W-questions
    if re.search(r"\b(who|what|where|when|why|how|which)\b", q):
        return "factual", 0.7

    # Default fallback
    return None, 0.0
    

def segment(query: str, chat_history: Optional[List[str]] = None) -> Dict[str, Any]:
    q = preprocess_query(query)
    slots = extract_slots(q, chat_history)

    # 1) rule-based quick checks (high precision)
    rule = rule_based_segment(q, chat_history)
    if rule:
        return {"segment": rule, "confidence": 0.99, "slots": slots, "route": f"{rule}_pipeline"}

    # 2) zero-shot classifier (if available)
    label, conf = zero_shot_segment(q)
    if label and conf >= 0.55:
        # normalize label 'factual_qa' -> 'factual' if model returns the old name
        if label == 'factual_qa':
            label = 'factual'
        return {"segment": label, "confidence": conf, "slots": slots, "route": f"{label}_pipeline"}

    # 3) fallback: use simple heuristic based on W-word
    if slots["w_words"]:
        return {"segment": "factual", "confidence": 0.5, "slots": slots, "route": "factual_pipeline"}

    # final fallback: default to factual for ambiguous/short queries
    return {"segment": "factual", "confidence": 0.4, "slots": slots, "route": "factual_pipeline"}

# quick CLI test
if __name__ == "__main__":
    examples = [
        "Who was Ashoka?",
        "Explain photosynthesis",
        "Solve 2x + 3 = 11",
        "Label the parts in the diagram of a flower",
        "Compare mitosis and meiosis",
        "Summarize chapter 1 of Beehive",
        "What about him?"
    ]
    for q in examples:
        print(q, "->", segment(q, chat_history=["Who is Margie?"]))