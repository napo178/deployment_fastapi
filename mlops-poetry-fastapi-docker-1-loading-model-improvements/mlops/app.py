from fastapi import FastAPI
from pydantic.tools import T
from mlops.routers import iris

app = FastAPI()
app.include_router(iris.router, prefix="/iris")

tags = ["test"]


@app.get("/", tags=tags)
async def root():
    return {"message": "Hello World"}


@app.get("/healthcheck", tags=tags, status_code=200)
async def healthcheck():
    return {"message": "Iris classifier is all ready to go!"}
