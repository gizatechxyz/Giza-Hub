from giza_actions.action import Action, action
from giza_actions.task import task
from giza_actions.model import GizaModel
import numpy as np

onnx_model_path = "ff_nn_compound_ur_prediction.onnx"
in_x = np.load("X_test_sample.npy")
model_input_2d = in_x.reshape(1, -1)  # Reshape to 2D array with 1 row


@task(name="Unverifiable Prediction with ONNX")
def prediction(model_input):
    model = GizaModel(model_path=onnx_model_path)
    result = model.predict(
        input_feed={model.session.get_inputs()[0].name: model_input}, verifiable=False
    )
    return result


@action(name="Unverifiable Execution: Prediction with ONNX", log_prints=True)
def execution():
    predicted_val = prediction(model_input_2d)
    print(f"Predicted val: {predicted_val}")
    return predicted_val


execution()
