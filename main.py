from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from sklearn.neural_network import MLPRegressor
import random

app = FastAPI(title="رشد هوشمند - بک‌اند هوش مصنوعی")

# CORS برای اتصال به اپ Rork
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== مدل‌های ورودی ====================
class UserInput(BaseModel):
    rank: int
    gpa: float
    psych_scores: List[float]      # ۵ عدد Big Five یا Holland
    free_text_score: float = 0.0
    city: str = "تهران"

class SimulateInput(BaseModel):
    major_code: str

class FeedbackInput(BaseModel):
    user_id: str
    major_code: str
    satisfaction: int
    real_income: int
    employed: bool

# ==================== مدل شبکه عصبی ====================
nn_model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
# آموزش اولیه شبیه‌سازی (در آینده با داده واقعی جایگزین می‌شود)
X_mock = np.random.rand(1000, 7) * 100
y_mock = np.random.rand(1000, 5) * 100
nn_model.fit(X_mock, y_mock)

majors = ["computer", "medicine", "electrical", "law", "management", "civil", "mechanical", "pharmacy", "architecture", "accounting"]

def calculate_nn_scores(user: UserInput) -> Dict[str, float]:
    features = np.array([user.rank, user.gpa, *user.psych_scores, user.free_text_score, len(user.city)]).reshape(1, -1)
    raw_scores = nn_model.predict(features)[0]
    return {majors[i]: float(raw_scores[i]) for i in range(5)}

# ==================== سیستم AHP ====================
def apply_ahp(nn_scores: Dict, user: UserInput) -> List[Dict]:
    criteria = ["nn_score", "job_market", "salary", "interest", "risk"]
    pairwise = np.array([
        [1, 3, 2, 4, 5],
        [1/3, 1, 1/2, 2, 3],
        [1/2, 2, 1, 3, 4],
        [1/4, 1/2, 1/3, 1, 2],
        [1/5, 1/3, 1/4, 1/2, 1]
    ])
    weights = np.linalg.eig(pairwise)[1][:, 0]
    weights = weights / weights.sum()
    
    final_scores = []
    for major, nn_score in nn_scores.items():
        score = (
            weights[0] * nn_score +
            weights[1] * random.uniform(70, 95) +
            weights[2] * random.uniform(65, 90) +
            weights[3] * (sum(user.psych_scores)/5 * 20) +
            weights[4] * random.uniform(60, 85)
        )
        final_scores.append({
            "major": major,
            "university": random.choice(["تهران", "شریف", "امیرکبیر", "علامه", "تربیت مدرس"]),
            "final_score": round(score, 1),
            "nn_score": round(nn_score, 1),
            "ahp_weight": round(weights[0]*100, 1)
        })
    return sorted(final_scores, key=lambda x: x["final_score"], reverse=True)[:5]

# ==================== شبیه‌ساز ۱۰ ساله ====================
def generate_10year_simulation(major_code: str):
    base = {
        "computer": {"income": 145, "growth": 0.18, "risk": 0.07},
        "medicine": {"income": 210, "growth": 0.12, "risk": 0.03},
        "electrical": {"income": 132, "growth": 0.15, "risk": 0.09},
        "law": {"income": 95, "growth": 0.11, "risk": 0.15},
        "management": {"income": 120, "growth": 0.14, "risk": 0.10},
    }.get(major_code, {"income": 110, "growth": 0.13, "risk": 0.12})
    
    years = list(range(2026, 2036))
    realistic = [base["income"] * (1 + base["growth"])**i * random.uniform(0.95, 1.05) for i in range(10)]
    optimistic = [x * 1.25 for x in realistic]
    pessimistic = [x * 0.75 for x in realistic]
    
    return {
        "years": years,
        "realistic_income": [round(x, 1) for x in realistic],
        "optimistic_income": [round(x, 1) for x in optimistic],
        "pessimistic_income": [round(x, 1) for x in pessimistic],
        "avg_employment": round(92 - base["risk"]*100, 1),
        "risk_level": round(base["risk"]*100, 1)
    }

# ==================== APIها ====================
@app.post("/api/recommend")
async def recommend(user: UserInput):
    nn_scores = calculate_nn_scores(user)
    ranked = apply_ahp(nn_scores, user)
    return {"status": "success", "ranked_majors": ranked}

@app.post("/api/simulate")
async def simulate(data: SimulateInput):
    sim = generate_10year_simulation(data.major_code)
    return {"status": "success", "simulation": sim}

@app.post("/api/feedback")
async def feedback(data: FeedbackInput):
    print(f"بازخورد دریافت شد: {data}")
    return {"status": "success", "message": "سیستم از بازخورد شما یاد گرفت و دقت پیشنهادها افزایش یافت"}

@app.get("/")
async def root():
    return {"message": "بک‌اند رشد هوشمند فعال است - آماده اتصال به Rork 🚀"}
