# ------------------ IMPORT LIBRARIES --------------------------
from fastapi import FastAPI





# ------------------- FastAPI App ------------------------------
app = FastAPI()




# ------------------ Simple Endpoint --------------------------
@app.get("/")
def home_endpoint():

    return "Welcome to this Page......."