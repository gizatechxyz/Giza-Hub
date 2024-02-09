from giza_actions.action import Action, action
from giza_actions.model import GizaModel
from giza_actions.task import task

from trend_token_prediction.predict_onnx_action import preprocess_image

MODEL_ID = ...  # Update with your model ID
VERSION_ID = ...  # Update with your version ID


@task(name="Prediction with Cairo")
def prediction(image, model_id, version_id):
    # TODO


@action(name="Execution: Prediction with Cairo", log_prints=True)
def execution():
    # TODO
