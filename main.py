from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np
import pandas as pd

# 1) إعداد FastAPI
app = FastAPI(title="Riyadh Aqar Price API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # لاحقاً ممكن تحددين دومين معين
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2) تحميل الباندل (المودل + الأنكودر + أسماء الأعمدة)
bundle = joblib.load("riyadh_aqar_xgb_bundle.pkl")
model = bundle["model"]
encoders = bundle["encoders"]
feature_names = bundle["feature_names"]

print("✅ Bundle loaded. Features:", feature_names)

# 3) قيم افتراضية لبقية الفيتشرز
# ملاحظة: غيّري الأرقام حسب القيم النموذجية في بياناتك (median / mode)
DEFAULT_VALUES = {
    'front':        'North',
    'rooms':        5,
    'lounges':      2,
    'bathrooms':    3,
    'street_width': 15,
    'stairs':       1,
    'property_age': 5,
    'driver_room':  0,
    'tent':         0,
    'patio':        0,
    'kitchen':      1,
    'outdoor_room': 0,
    'garage':       1,
    'duplex':       0,
    'space':        300,
    'apartments':   0,
    'maid_room':    0,
    'elevator':     0,
    'furnished':    0,
    'pool':         0,
    'basement':     0,
    'neighbourhood':'Akaz',
    'location':     'South Riyadh',
    'space_log':    np.log1p(300),
}

CATEGORICAL_COLS = ['front', 'neighbourhood', 'location']


# 4) شكل البيانات التي يستقبلها الـ API من الواجهة
class VillaInput(BaseModel):
    space: float
    rooms: int
    bathrooms: int
    street_width: float
    property_age: int
    front: str
    neighbourhood: str
    location: str


# 5) دالة تبني الـ DataFrame بنفس شكل التدريب
def build_features(user: VillaInput) -> pd.DataFrame:
    values = DEFAULT_VALUES.copy()

    # نحدّث القيم التي يدخلها اليوزر
    values['space']        = user.space
    values['space_log']    = float(np.log1p(user.space))
    values['rooms']        = user.rooms
    values['bathrooms']    = user.bathrooms
    values['street_width'] = user.street_width
    values['property_age'] = user.property_age
    values['front']        = user.front
    values['neighbourhood']= user.neighbourhood
    values['location']     = user.location

    df = pd.DataFrame([values])

    # نطبّق نفس الـ LabelEncoders التي استخدمناها في التدريب
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        df[col] = le.transform(df[col])

    # نتأكد من نفس ترتيب الأعمدة
    df = df[feature_names]
    return df


# 6) Endpoints
@app.get("/")
def root():
    return {"message": "Riyadh Aqar Price API is running 👋"}


@app.post("/predict_price")
def predict_price(villa: VillaInput):
    X = build_features(villa)

    # المودل يتنبأ بالـ log(price)
    y_log_pred = model.predict(X)[0]

    # نحول من log(price) إلى سعر حقيقي
    price_pred = float(np.expm1(y_log_pred))   # لأنك استخدمت log1p على الأغلب

    return {
        "predicted_price": round(price_pred, 2),
        "currency": "SAR",
        "log_prediction": float(y_log_pred),
        "input_used": villa.dict()
    }
