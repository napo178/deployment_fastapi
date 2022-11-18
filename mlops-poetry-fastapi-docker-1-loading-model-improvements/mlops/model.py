from pathlib import Path
import numpy as np
import joblib

from mlops.schemas.iris import IrisInput, IrisResponse


class IrisClassifier:
    def __init__(self):
        self.clf = self.load_model()
        self.iris_type = {0: "setosa", 1: "versicolor", 2: "virginica"}

    def load_model(self):

        """
        Loads a provided model in pickle format.

        Returns:
            The trained model.
        """

        cwd = Path.cwd()
        model_path = cwd / "mlops" / "model.pickle"

        return joblib.load(model_path)

    def classify_iris(self, features: IrisInput) -> dict:

        """
        Runs the predict method of the model giving the results.

        Returns:
            IrisResponse: Type and probability.
        """
        prediction = self.clf.predict_proba(
            [[features.sepal_l, features.sepal_w, features.petal_l, features.petal_w]]
        )
        return IrisResponse(
            type=self.iris_type[np.argmax(prediction)],
            probability=round(max(prediction[0]), 2),
        )


if __name__ == "__main__":

    """
    Just for testing purposes.
    """

    model = IrisClassifier()
    prediction = model.classify_iris(
        {"sepal_l": 1.1, "sepal_w": 2.1, "petal_l": 3.1, "petal_w": 4.1}
    )
    print(prediction)
