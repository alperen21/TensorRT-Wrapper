import torch_tensorrt
import torch
import pandas as pd

class TensorRTWrapper():
  def __init__(self, pytorch_model : torch.nn.Module, input_dimensions = list[int]):
    traced_model = torch.jit.trace(pytorch_model, [torch.randn(input_dimensions).to("cuda")])
    self.model = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(input_dimensions, dtype=torch.float32)],
        enabled_precisions={torch.float32},
        truncate_long_and_double=True
    )

  def predict(self, img):
    img_batch = torch.unsqueeze(img, 0)
    img_batch.shape
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(img_batch)
    return outputs
