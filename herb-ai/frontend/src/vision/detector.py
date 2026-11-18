import base64
import io
import os
import sys

import httpx
import open_clip
import torch
from database.db_manager import query_visual_memory, store_visual_memory
from dotenv import load_dotenv
from google import genai
from PIL import Image
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))

app_root = os.path.abspath(os.path.join(project_root, "../"))
if app_root not in sys.path:
    sys.path.insert(0, app_root)

from frontend.src.rag.know_gen import AutoKnowledgeGenerator
from frontend.src.rag.vector_store import LocalVectorStoreEngine


class BotanicalDetector:
    def __init__(self, model_path: str = "herb-ai/best.pt"):
        print(f"Loading Edge YOLO model: {model_path}...")
        self.model = YOLO(model_path)

        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = (
            genai.Client(api_key=self.api_key) if self.api_key else None
        )

        print("🧠 Loading OpenCLIP visual encoder...")
        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
        )
        self.clip_model.eval()

    def _get_image_vector(self, pil_image: Image.Image) -> list:
        image_tensor = self.clip_preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().tolist()[0]

    def analyze_image(
        self, image_bytes: bytes, session_id: str = "default_session"
    ) -> list:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((640, 640))
        image_vector = self._get_image_vector(image)

        results_list = []

        # 1. CLOUD VECTOR MEMORY RECALL
        try:
            rows = query_visual_memory(image_vector, limit=1)
            if rows:
                learned_name, sid, dist = rows[0]
                if dist < 0.25:
                    print(
                        f"⚡ Visual Memory Recall: {learned_name} (Distance: {dist:.3f})"
                    )
                    return [{"predicted_class": learned_name, "confidence": 0.98}]
        except Exception as mem_err:
            pass

        temp_target = "/tmp/temp_inference_target.jpg"
        image.save(temp_target)

        results = self.model(temp_target, verbose=False)
        top_idx = results[0].probs.top1
        predicted_class = results[0].names[top_idx]
        confidence = float(results[0].probs.top1conf)

        if confidence < 0.70:
            print(
                "🔍 YOLO confidence low. Attempting Cloud Vision for multiple plants..."
            )
            try:
                if not self.gemini_client:
                    raise ValueError("Gemini Client not initialized.")

                prompt = (
                    "Identify all botanical or medicinal substances visible in this image. "
                    "Reply ONLY with a comma-separated list of the common botanical or TCM names (e.g., 'Rosemary, Thyme, Mint'). "
                    "If you cannot identify any, reply EXACTLY with 'Unidentified Anomaly'."
                )
                active_gemini_models = [
                    "gemini-3.6-flash",
                    "gemini-3.5-flash",
                    "gemini-3.1-flash-lite",
                    "gemini-2.5-flash",
                ]

                cloud_success = False
                for gemini_model in active_gemini_models:
                    try:
                        response = self.gemini_client.models.generate_content(
                            model=gemini_model, contents=[prompt, image]
                        )
                        discovered_text = response.text.strip()

                        if "Unidentified" in discovered_text or not discovered_text:
                            results_list.append(
                                {
                                    "predicted_class": "Unidentified Anomaly",
                                    "confidence": 0.0,
                                }
                            )
                        else:
                            plant_names = [
                                name.strip()
                                for name in discovered_text.split(",")
                                if name.strip()
                            ]
                            for name in plant_names:
                                print(
                                    f"☁️ Cloud Vision Found: {name} via {gemini_model}"
                                )
                                results_list.append(
                                    {"predicted_class": name, "confidence": 0.95}
                                )

                        cloud_success = True
                        break
                    except Exception as model_error:
                        continue

                if not cloud_success:
                    raise RuntimeError("All Gemini fallbacks exhausted.")

            except Exception as cloud_error:
                print(
                    f"☁️ Cloud Vision Exhausted ({cloud_error}). Auto-failing over to Edge Vision (Ollama)..."
                )

                buffered = io.BytesIO()
                image.thumbnail((512, 512))
                image.save(buffered, format="JPEG", quality=85)
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

                try:
                    with httpx.Client() as client:
                        resp = client.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": "llava",
                                "prompt": (
                                    "Identify all botanical or medicinal substances visible in this image. "
                                    "Reply ONLY with a comma-separated list of the common botanical or TCM names (e.g., 'Rosemary, Thyme'). "
                                    "If you cannot identify any, reply EXACTLY with 'Unidentified Anomaly'."
                                ),
                                "images": [base64_image],
                                "stream": False,
                            },
                            timeout=120.0,
                        )
                        if resp.status_code == 200:
                            discovered_text = resp.json().get("response", "").strip()
                            if not discovered_text or "Unidentified" in discovered_text:
                                results_list.append(
                                    {
                                        "predicted_class": "Unidentified Anomaly",
                                        "confidence": 0.0,
                                    }
                                )
                            else:
                                plant_names = [
                                    name.strip()
                                    for name in discovered_text.split(",")
                                    if name.strip()
                                ]
                                for name in plant_names:
                                    print(f"🔌 Edge Vision Found: {name}")
                                    results_list.append(
                                        {"predicted_class": name, "confidence": 0.85}
                                    )
                except Exception as e:
                    print(f"❌ Inferences failed: {e}")
                    results_list.append(
                        {"predicted_class": "Unidentified Anomaly", "confidence": 0.0}
                    )
        else:
            results_list.append(
                {"predicted_class": predicted_class, "confidence": round(confidence, 2)}
            )

        for res in results_list:
            p_class = res["predicted_class"]
            if res["confidence"] > 0.70 and p_class != "Unidentified Anomaly":
                try:
                    mem_id = f"img_{p_class}_{os.urandom(4).hex()}"
                    store_visual_memory(mem_id, p_class, session_id, image_vector)
                except Exception:
                    pass

        if os.path.exists(temp_target):
            os.remove(temp_target)

        return results_list
