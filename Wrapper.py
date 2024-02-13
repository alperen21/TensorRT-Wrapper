import torch_tensorrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os

class TensorrtModel(nn.Module):
  def __init__(self, pytorch_model_weights, input_dimensions : tuple[int] = (32, 3, 224, 224), replace = False):
    nn.Module.__init__(self)
    self.__file_name = None
    self.__tensorrt_engine = None

    self.__override_torch_save()
    if not isinstance(pytorch_model_weights, str):
      pytorch_model = pytorch_model_weights
      traced_model = torch.jit.trace(pytorch_model, [torch.randn(input_dimensions).to("cuda")])
      self.__tensorrt_engine = torch_tensorrt.compile(
          traced_model,
          inputs=[torch_tensorrt.Input(input_dimensions, dtype=torch.float32)],
          enabled_precisions={torch.float32},
          truncate_long_and_double=True
      )
      return


    self.__file_name, _ = os.path.splitext(pytorch_model_weights)
    if os.path.exists(self.__file_name + ".ts") and not replace:
      self.__model = torch.jit.load(self.__file_name + ".ts")
      return
    else:
      print("Not using saved tensorrt model")

    pytorch_model = torch.load(pytorch_model_weights)
    traced_model = torch.jit.trace(pytorch_model, [torch.randn(input_dimensions).to("cuda")])
    self.__tensorrt_engine = torch_tensorrt.compile(
        traced_model,
        inputs=[torch_tensorrt.Input(input_dimensions, dtype=torch.float32)],
        enabled_precisions={torch.float32},
        truncate_long_and_double=True
    )

    
  
  def __get_engine(self):
    return self.__tensorrt_engine

  
  def __override_torch_save(self):
    import types
    old_save_function = types.FunctionType(torch.save.__code__, torch.save.__globals__, name=torch.save.__name__,
                                  argdefs=torch.save.__defaults__, closure=torch.save.__closure__)
    
    def overridden_save(obj, f, pickle_module=torch.serialization.pickle, pickle_protocol=torch.serialization.DEFAULT_PROTOCOL):
      if isinstance(obj, TensorrtModel):
        torch.jit.save(obj.__get_engine(), f)
      else:
        old_save_function(obj, f, pickle_module=pickle_module, pickle_protocol=pickle_protocol)

    torch.save = overridden_save


  def forward(self, img):
    img_batch = torch.unsqueeze(img, 0)
    img_batch.shape
    outputs = self.__tensorrt_engine(img_batch)
    return outputs

