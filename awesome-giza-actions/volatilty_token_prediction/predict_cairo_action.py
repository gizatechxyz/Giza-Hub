from giza_actions.action import Action, action
from giza_actions.model import GizaModel
from giza_actions.task import task
import polars as pl
import numpy as np
import torch


@task(name="Prediction with Cairo")
def prediction(X_test, model):
    (result, request_id) = model.predict(
        input_feed={"input_feed":X_test}, verifiable=True, output_dtype="arr_fixed_point"
    )
    return result, request_id

@action(name="Execution: Prediction with Cairo", log_prints=True)
def execution():
    model = GizaModel(id=293, version=1)
    df = pl.read_csv("./example_token_trend.csv")
    model_input = df.to_numpy().astype(np.float32)
    (result, request_id) = prediction(model_input, model)
    return result, request_id
(result, request_id) = execution()
print(f"Result: {result}, Request ID: {request_id}")
