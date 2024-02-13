from giza_actions.action import Action, action
from giza_actions.task import task
from giza_actions.model import GizaModel
import numpy as np

in_x = np.load("X_test_sample.npy")
model_input_2d = in_x.reshape(1, -1)  # Reshape to 2D array with 1 row
model_id = 274
version_id = 7


@task(name="Verifiable Prediction with Cairo")
def prediction(model_input, model_id, version_id):
    # Initialize a GizaModel with model and version id.
    model = GizaModel(id=model_id, version=version_id)

    # Call the predict function.
    # Set `verifiable` to True, and define the expecting output datatype.
    (result, request_id) = model.predict(
        input_feed={"model_input": model_input},
        verifiable=True,
        output_dtype="arr_fixed_point",
    )
    return result, request_id


@action(name="Verifiable Execution: Prediction with Cairo", log_prints=True)
def execution():
    (result, request_id) = prediction(model_input_2d, model_id, version_id)
    return result, request_id


result, request_id = execution()
print(f"Result: {result}, Request ID: {request_id}")
