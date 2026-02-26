import os
import json
import random
from typing import List, Dict
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ==================== تنظیمات پایه و امنیت ====================
app = FastAPI(
    title="رشد هوشمند - بک‌اند هوش مصنوعی",
    description="سیستم هوشمند پیشنهاد رشته تحصیلی و شبیه‌سازی مسیر شغلی",
    version="2.0.0"
)

# ۵. اصلاح CORS: استفاده از لیست مجاز به جای دسترسی باز (*)
# می‌توانید آدرس‌های مجاز را از طریق متغیر محیطی تنظیم کنید
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", 
    "http://localhost:3000,http://localhost:8080,https://app.rork.ir"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ==================== مدل‌های اعتبارسنجی (Pydantic) ====================
class UserInput(BaseModel):
    rank: int = Field(..., ge=1, description="رتبه کنکور")
    gpa: float = Field(..., ge=0, le=20, description="معدل")
    psych_scores: List[float] = Field(..., min_length=5, max_length=5, description="۵ نمره آزمون روان‌شناختی")
    free_text_score: float = Field(default=0.0, ge=0, le=100)
    city: str = Field(default="تهران", min_length=2)

class SimulateInput(BaseModel):
    major_code: str

class FeedbackInput(BaseModel):
    user_id: str
    major_code: str
    satisfaction: int = Field(..., ge=1, le=5, description="رضایت از ۱ تا ۵")
    real_income: int = Field(..., ge=0, description="درآمد واقعی به میلیون تومان")
    employed: bool

# ==================== تنظیمات مدل و داده‌های پایه ====================
MAJORS = [
    "computer", "medicine", "electrical", "law", "management", 
    "civil", "mechanical", "pharmacy", "architecture", "accounting"
]
NUM_FEATURES = 9  # (rank, gpa, 5x psych, free_text, city_length)

# ۳. اصلاح مقیاس‌بندی: استفاده از StandardScaler
scaler = StandardScaler()
nn_model = MLPRegressor(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)

# ۱. اصلاح ابعاد: تولید داده‌های Mock منطبق با ۹ ویژگی ورودی و ۱۰ رشته خروجی
def initialize_model():
    # تولید داده‌های شبیه‌سازی شده منطقی‌تر برای آموزش اولیه
    mock_rank = np.random.uniform(1, 100000, 1000)
    mock_gpa = np.random.uniform(10, 20, 1000)
    mock_psych = np.random.uniform(0, 100, (1000, 5))
    mock_text = np.random.uniform(0, 100, 1000)
    mock_city_len = np.random.uniform(3, 15, 1000)
    
    # ترکیب ویژگی‌ها (ماتریس 1000x9)
    X_mock = np.column_stack((mock_rank, mock_gpa, mock_psych, mock_text, mock_city_len))
    # خروجی برای ۱۰ رشته (ماتریس 1000x10)
    y_mock = np.random.rand(1000, len(MAJORS)) * 100
    
    # فیت کردن Scaler و مدل شبکه عصبی
    X_scaled = scaler.fit_transform(X_mock)
    nn_model.fit(X_scaled, y_mock)

initialize_model()

def calculate_nn_scores(user: UserInput) -> Dict[str, float]:
    features = np.array([
        user.rank, 
        user.gpa, 
        *user.psych_scores, 
        user.free_text_score, 
        len(user.city)
    ]).reshape(1, -1)
    
    # حتماً باید داده ورودی کاربر هم نرمال‌سازی شود
    features_scaled = scaler.transform(features)
    raw_scores = nn_model.predict(features_scaled)[0]
    
    return {MAJORS[i]: float(raw_scores[i]) for i in range(len(MAJORS))}

# ==================== سیستم AHP ====================
# ۲. اصلاح AHP: محاسبه دقیق وزن‌ها بدون درگیری با اعداد مختلط
def _compute_ahp_weights() -> np.ndarray:
    pairwise = np.array([
        [1,   3,   2,   4,   5  ],
        [1/3, 1,   1/2, 2,   3  ],
        [1/2, 2,   1,   3,   4  ],
        [1/4, 1/2, 1/3, 1,   2  ],
        [1/5, 1/3, 1/4, 1/2, 1  ]
    ])
    eigenvalues, eigenvectors = np.linalg.eig(pairwise)
    
    # پیدا کردن ایندکس بزرگترین مقدار ویژه حقیقی
    max_idx = np.argmax(eigenvalues.real)
    # استخراج بردار ویژه مربوطه و گرفتن قدر مطلق بخش حقیقی آن
    weights = np.abs(eigenvectors[:, max_idx].real)
    # نرمال‌سازی (مجموع وزن‌ها = ۱)
    return weights / weights.sum()

# محاسبه وزن‌ها فقط یکبار هنگام اجرای سرور
AHP_WEIGHTS = _compute_ahp_weights()

def apply_ahp(nn_scores: Dict[str, float], user: UserInput) -> List[Dict]:
    w = AHP_WEIGHTS
    avg_psych = sum(user.psych_scores) / len(user.psych_scores)
    
    final_scores = []
    for major, nn_score in nn_scores.items():
        score = float(
            w[0] * nn_score +
            w[1] * random.uniform(70, 95) +
            w[2] * random.uniform(65, 90) +
            w[3] * (avg_psych * 20) +
            w[4] * random.uniform(60, 85)
        )
        final_scores.append({
            "major": major,
            "university": random.choice(["تهران", "شریف", "امیرکبیر", "علامه", "تربیت مدرس"]),
            "final_score": round(score, 1),
            "nn_score": round(nn_score, 1),
            "ahp_weight": round(float(w[0]) * 100, 1)
        })
        
    return sorted(final_scores, key=lambda x: x["final_score"], reverse=True)[:5]

# ==================== شبیه‌ساز ۱۰ ساله ====================
MAJOR_BASE_STATS = {
    "computer": {"income": 145, "growth": 0.18, "risk": 0.07},
    "medicine": {"income": 210, "growth": 0.12, "risk": 0.03},
    "electrical": {"income": 132, "growth": 0.15, "risk": 0.09},
    "law": {"income": 95, "growth": 0.11, "risk": 0.15},
    "management": {"income": 120, "growth": 0.14, "risk": 0.10},
    "civil": {"income": 105, "growth": 0.10, "risk": 0.14},
    "mechanical": {"income": 115, "growth": 0.12, "risk": 0.11},
    "pharmacy": {"income": 160, "growth": 0.11, "risk": 0.05},
    "architecture": {"income": 100, "growth": 0.09, "risk": 0.16},
    "accounting": {"income": 110, "growth": 0.13, "risk": 0.08},
}

def generate_10year_simulation(major_code: str):
    if major_code not in MAJOR_BASE_STATS:
        raise HTTPException(status_code=404, detail="رشته مورد نظر یافت نشد")
        
    base = MAJOR_BASE_STATS[major_code]
    years = list(range(2026, 2036))
    
    realistic = [base["income"] * (1 + base["growth"])**i * random.uniform(0.95, 1.05) for i in range(10)]
    optimistic = [x * 1.25 for x in realistic]
    pessimistic = [x * 0.75 for x in realistic]
    
    return {
        "years": years,
        "realistic_income": [round(x, 1) for x in realistic],
        "optimistic_income": [round(x, 1) for x in optimistic],
        "pessimistic_income": [round(x, 1) for x in pessimistic],
        "avg_employment": round(92 - base["risk"] * 100, 1),
        "risk_level": round(base["risk"] * 100, 1)
    }

# ==================== سیستم مدیریت بازخورد ====================
FEEDBACK_FILE = Path("feedback_data.json")

# ۴. اصلاح سیستم فیدبک: ذخیره‌سازی واقعی داده‌ها
def save_feedback_to_db(data: FeedbackInput):
    feedbacks = []
    if FEEDBACK_FILE.exists():
        try:
            feedbacks = json.loads(FEEDBACK_FILE.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            pass
            
    feedbacks.append(data.model_dump())
    FEEDBACK_FILE.write_text(json.dumps(feedbacks, ensure_ascii=False, indent=2), encoding='utf-8')
    return len(feedbacks)

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
    total_records = save_feedback_to_db(data)
    return {
        "status": "success", 
        "message": "بازخورد با موفقیت ذخیره شد.",
        "total_feedbacks": total_records
    }

@app.get("/")
async def root():
    return {"message": "بک‌اند رشد هوشمند فعال و در حال اجراست 🚀", "status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # غیرفعال کردن لاگ‌های اضافه در محیط پروداکشن
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
