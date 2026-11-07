# herb-ai/frontend/src/rag/omniroute.py
import numpy as np
from sentence_transformers import SentenceTransformer

class OmniRouter:
    def __init__(self, embedding_model: SentenceTransformer):
        self.model = embedding_model
        
        # Define semantic trigger phrases for each route
        self.routes = {
            "vision_telemetry": [
                "what plant am i looking at",
                "what did the camera just see",
                "identify the herb on my screen",
                "what is the name of the photo i uploaded",
                "did you detect any anomalies"
            ],
            "clinical_rag": [
                "what are the side effects of",
                "how do I prepare",
                "what is the dosage for",
                "tell me the medicinal properties of",
                "is this toxic or safe"
            ],
            "system_command": [
                "clear my session",
                "wipe my memory",
                "reset the dashboard"
            ]
        }
        
        print("🛣️ [OMNIROUTE] Pre-computing route embeddings...")
        self.route_embeddings = {
            route: self.model.encode(phrases) 
            for route, phrases in self.routes.items()
        }

    def cosine_similarity(self, a, b):
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))

    def route_query(self, user_query: str) -> str:
        """Determines the correct pipeline route based on semantic proximity."""
        query_vec = self.model.encode([user_query.lower()])[0]
        
        best_route = "general_chat" # Default fallback
        highest_score = 0.0
        
        for route_name, phrase_vectors in self.route_embeddings.items():
            # Check similarity against all trigger phrases for this route
            similarities = self.cosine_similarity(query_vec, phrase_vectors)
            max_sim = np.max(similarities)
            
            if max_sim > highest_score and max_sim > 0.65: # 0.65 is the confidence threshold
                highest_score = max_sim
                best_route = route_name
                
        print(f"🔀 [OMNIROUTE] Navigating query to: {best_route.upper()} (Confidence: {highest_score:.2f})")
        return best_route