from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from starlette.responses import JSONResponse

from mlops.model import IrisClassifier
from mlops.schemas.iris import IrisInput

router = APIRouter()


@router.post("/classify_iris")
def classify_iris(iris_features: IrisInput) -> JSONResponse:

    """
    Loads the Iris classifier and predict the type.

    Args:
        iris_features(IrisInput): Info about the flower.

    Returns:
        JSONResponse: The predicted type and probability

    Raises:
        HTTPException: If there is problems with Iris or data.
    """

    try:
        iris_classifier = IrisClassifier()
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {err}")

    try:
        prediction = iris_classifier.classify_iris(iris_features)
    except ValidationError as err:
        raise HTTPException(
            status_code=500, detail=f"See if all fields are OK, ValidationError: {err}"
        )

    return JSONResponse(prediction.dict())
