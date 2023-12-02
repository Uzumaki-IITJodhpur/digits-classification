import unittest
import joblib
from sklearn.linear_model import LogisticRegression

class TestLogisticRegressionModel(unittest.TestCase):

    def test_model_type(self):

        model = joblib.load('./m22aie245lrlbfgs.joblib')  # replace with your actual model path
        self.assertIsInstance(model, LogisticRegression, "Loaded model is not a Logistic Regression model")


class TestLogisticRegressionModel(unittest.TestCase):

    def test_solver_name(self):
        solver_name = "liblinear"  # replace with the expected solver name
        model = joblib.load('./m22aie245lrliblinear.joblib')  # replace with your actual model path
        self.assertEqual(model.get_params()['solver'], solver_name, f"Solver name in the model does not match '{solver_name}'")