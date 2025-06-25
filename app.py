from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json, pickle, os
import google.generativeai as genai

app = Flask(__name__)

# Load FAISS index and metadata
index = faiss.read_index("faiss.index")
with open("metadata.pkl", "rb") as f:
    data = pickle.load(f)

corpus = data["corpus"]
exercise_data = data["exercise_data"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API key
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))  # Secrets injected by HF
llm = genai.GenerativeModel("gemini-2.0-flash-lite")  # Free-tier optimized

@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    try:
        data = request.json
        query = (
            f"{data['fitness_goal']} exercises for {data['gender']} with {data['available_equipment']}"
        )
        qvec = model.encode([query])
        D, I = index.search(np.array(qvec), 15)
        selected = [corpus[i] for i in I[0]]

        prompt = f"""
You're a certified fitness coach. Based on the user input:
- Age: {data['age']}, Gender: {data['gender']}
- Goal: {data['fitness_goal']}, Experience: {data['experience_level']}
- Equipment: {data['available_equipment']}, Health: {data['health_conditions']}

Create a 7-day workout plan using the exercises below. Each day includes 4–5 exercises.

{selected}

Return ONLY a valid JSON array:
[
  {{
    "day": "Day 1",
    "exercises": [
      {{
        "name": "", "target_muscle": "", "description": "", "difficulty": "",
        "type": "", "image_url": "", "video_url": "", "reps": ""
      }}
    ]
  }}
]
"""
        response = llm.generate_content(prompt)
        try:
            return jsonify({"plan": json.loads(response.text)})
        except:
            return jsonify({"raw_output": response.text, "error": "Invalid JSON"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "HealthyEats Fitness Plan API is live!"

# ✅ Required for Hugging Face Spaces to detect and bind port 7860
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
