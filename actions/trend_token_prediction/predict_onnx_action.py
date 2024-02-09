import numpy as np
import torch
import torch.nn.functional as F
from giza_actions.action import Action, action
from giza_actions.model import GizaModel
from giza_actions.task import task
from PIL import Image


@task(name="Preprocess Image")
def preprocess_image(image_path):
    # Load image, convert to grayscale, resize and normalize
    image = Image.open(image_path).convert("L")
    # Resize to match the input size of the model
    image = image.resize((14, 14))
    image = np.array(image).astype("float32") / 255
    image = image.reshape(1, 196)  # Reshape to (1, 196) for model input
    return image


@task(name="Prediction with ONNX")
def prediction(image):
    model = GizaModel(model_path="./mnist_model.onnx")

    result = model.predict(input_feed={"onnx::Gemm_0": image}, verifiable=False)

    # Convert result to a PyTorch tensor
    result_tensor = torch.tensor(result)
    # Apply softmax to convert to probabilities
    probabilities = F.softmax(result_tensor, dim=1)
    # Use argmax to get the predicted class
    predicted_class = torch.argmax(probabilities, dim=1)

    return predicted_class.item()


@action(name="Execution: Prediction with ONNX", log_prints=True)
def execution():
    image = preprocess_image("./zero.jpg")
    predicted_digit = prediction(image)
    print(f"Predicted Digit: {predicted_digit}")

    return predicted_digit


if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-mnist-predict-action")
    action_deploy.serve(name="pytorch-mnist-predict-deployment")
