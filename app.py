from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import os
import shutil

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    hospital_name = Column(String, default="")
    is_active = Column(Boolean, default=True)

class Donor(Base):
    __tablename__ = "donors"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    age = Column(Integer)
    bmi = Column(Float, default=22.0)
    gender = Column(String)
    health_status = Column(String)
    beneficial_genera = Column(String, default="")
    shannon_index = Column(Float)
    has_pathogen = Column(Boolean, default=False)
    enterotype = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer)

class Sample(Base):
    __tablename__ = "samples"
    id = Column(Integer, primary_key=True)
    donor_id = Column(Integer)
    sample_type = Column(String)
    abundance_json = Column(JSON)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def shannon_index(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def braycurtis(u, v):
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    return np.sum(np.abs(u - v)) / np.sum(u + v)

def parse_csv_to_series(file_path):
    df = pd.read_csv(file_path, index_col=0)
    col = 'abundance' if 'abundance' in df.columns else df.columns[0]
    s = df[col].fillna(0)
    s = s / s.sum()
    return s

security = HTTPBearer()

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401)
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401)
    return user

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return FileResponse("login.html")

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return FileResponse("index.html")

@app.post("/api/auth/login")
def login_or_register(data: dict, db: Session = Depends(get_db)):
    username = data.get("username")
    password = data.get("password")
    license_key = data.get("license_key", "")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        valid_keys = ["FMT2025", "HOSPITAL001", "TRIAL"]
        if license_key not in valid_keys:
            return {"error": "无效的激活码"}
        hashed_pwd = get_password_hash(password)
        user = User(username=username, hashed_password=hashed_pwd)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if not verify_password(password, user.hashed_password):
            return {"error": "密码错误"}
    token = create_access_token(data={"sub": user.username})
    return {"token": token, "username": user.username}

@app.post("/donors/upload")
async def upload_donor(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    bmi: float = Form(22.0),
    gender: str = Form(...),
    health: str = Form(...),
    beneficial_genera: str = Form(""),
    has_pathogen: bool = Form(False),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    tmp_path = f"temp_{name}.csv"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    series = parse_csv_to_series(tmp_path)
    shannon = shannon_index(series.values)
    abund_dict = series.to_dict()
    donor = Donor(
        name=name, age=age, bmi=bmi, gender=gender, health_status=health,
        beneficial_genera=beneficial_genera,
        shannon_index=shannon, has_pathogen=has_pathogen,
        user_id=user.id
    )
    db.add(donor)
    db.flush()
    sample = Sample(donor_id=donor.id, sample_type="donor", abundance_json=json.dumps(abund_dict))
    db.add(sample)
    db.commit()
    os.remove(tmp_path)
    return {"message": f"Donor {name} saved", "donor_id": donor.id, "shannon": round(shannon, 3)}

@app.get("/donors")
def list_donors(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    donors = db.query(Donor).filter(Donor.user_id == user.id).all()
    return [{"id": d.id, "name": d.name, "age": d.age, "bmi": d.bmi, "gender": d.gender,
             "shannon": round(d.shannon_index, 2) if d.shannon_index else None,
             "has_pathogen": d.has_pathogen, "beneficial_genera": d.beneficial_genera} for d in donors]

@app.post("/match")
async def match_recipient(
    file: UploadFile = File(...),
    indication: str = Form("general"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    tmp_path = "temp_recipient.csv"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    rec_series = parse_csv_to_series(tmp_path)
    rec_dict = rec_series.to_dict()
    rec_shannon = shannon_index(rec_series.values)

    weight_map = {
        "uc": (0.7, 0.3),
        "obesity": (0.3, 0.7),
        "cirrhosis": (0.5, 0.5),
        "immune_skin": (0.4, 0.6),
        "general": (0.5, 0.5)
    }
    w_sim, w_div = weight_map.get(indication, (0.5, 0.5))

    donors = db.query(Donor).filter(Donor.user_id == user.id, Donor.has_pathogen == False).all()
    results = []
    for d in donors:
        sample = db.query(Sample).filter(Sample.donor_id == d.id, Sample.sample_type == "donor").first()
        if not sample:
            continue
        don_dict = json.loads(sample.abundance_json)
        all_genera = sorted(set(rec_dict.keys()) | set(don_dict.keys()))
        v1 = np.array([rec_dict.get(g, 0) for g in all_genera])
        v2 = np.array([don_dict.get(g, 0) for g in all_genera])
        bc = braycurtis(v1, v2)
        sim_score = 1 - bc
        if rec_shannon > 0:
            div_advantage = min(1.0, d.shannon_index / rec_shannon)
        else:
            div_advantage = 1.0
        total_score = w_sim * sim_score + w_div * div_advantage

        if d.beneficial_genera:
            beneficial_list = [g.strip() for g in d.beneficial_genera.split(",")]
            beneficial_abund = sum(don_dict.get(g, 0) for g in beneficial_list)
        else:
            beneficial_abund = None

        results.append({
            "donor_id": d.id, "name": d.name, "age": d.age, "bmi": d.bmi,
            "braycurtis": round(bc, 4), "sim_score": round(sim_score, 4),
            "div_score": round(div_advantage, 4), "total_score": round(total_score, 4),
            "shannon_donor": d.shannon_index, "shannon_recipient": round(rec_shannon, 4),
            "beneficial_abund": round(beneficial_abund, 4) if beneficial_abund is not None else None,
            "health_status": d.health_status
        })

    if not results:
        os.remove(tmp_path)
        return {"error": "没有可选供体"}

    results.sort(key=lambda x: x["total_score"], reverse=True)
    best = results[0]
    if best["total_score"] >= 0.75:
        level = "⭐ 强烈推荐"
    elif best["total_score"] >= 0.5:
        level = "✔ 可考虑"
    else:
        level = "⚠ 需进一步评估"

    os.remove(tmp_path)
    return {
        "indication": indication,
        "weights": {"similarity": w_sim, "diversity": w_div},
        "best_match": best,
        "recommend_level": level,
        "alternatives": results[1:4]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)