from pydantic import BaseModel


class IrisInput(BaseModel):
    sepal_l: float
    sepal_w: float
    petal_l: float
    petal_w: float


class IrisResponse(BaseModel):
    type: str
    probability: float
