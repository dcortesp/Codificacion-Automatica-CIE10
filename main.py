from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import get_code, translate, traducir, Description

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return "Welcome to clinical ICD-10 code prediction!"

@app.get("/icd10")
def get_description(text = None):
    if len(text) > 0:
        text_en = traducir(text)
        codes = get_code(text_en)
        return {'ICD-10_Code': codes}
    else:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE, detail="Empty descriptions not acceptable...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)