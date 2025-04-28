from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.optimizer import simplex_solver, solve_graphic
#Oi sou gustavo
app = FastAPI()


origins = [
    "http://localhost:4200",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def handler(request):
    return {
        "statusCode": 200,
        "body": "Hello, this is the main.py function running on Vercel!"
    }

@app.get("/")
def read_root():
    return {"message": "Sistema de Otimização Rodando"}

@app.post("/simplex")
def simplex_endpoint(problem: dict):
    result = simplex_solver(problem)
    return result

@app.post("/grafico")
def grafico_endpoint(problem: dict):
    return solve_graphic(problem)

