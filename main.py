# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import scipy.sparse
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

base_path = os.path.dirname(__file__)
model_dir = os.path.join(base_path, "model")

with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

X = scipy.sparse.load_npz(os.path.join(model_dir, "tfidf_matrix.npz"))
df = pd.read_csv(os.path.join(model_dir, "books_cleaned.csv"))

app = FastAPI()

origins = [
    "https://bookify-pearl.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    query: str
    top_n: int = 5

@app.get("/")
def server_listen():
    return {"message":"Hello from Server!!"}

@app.post("/recommend")
def recommend_books(input: InputText):
    query_vec = vectorizer.transform([input.query])
    sims = (X @ query_vec.T).toarray().ravel()
    top_indices = np.argsort(sims)[-input.top_n:][::-1]
    results = df.iloc[top_indices].to_dict(orient="records")
    return {"recommendations": results}

@app.get("/getRecommendBook/{isbn13}")
def get_book_by_isbn(isbn13: str):
    try:
        isbn13_float = float(isbn13)
        book = df[df['isbn13'] == isbn13_float]
        if book.empty:
            return {"message": "Book not found"}
        return book.iloc[0].to_dict()
    except ValueError:
        return {"message": "Invalid ISBN format"}
