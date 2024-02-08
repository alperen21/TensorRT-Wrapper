from Wrapper import TensorRTWrapper

import os
from PIL import Image
import ultralytics
import torch
import torchvision
import tensorrt as trt
from torchvision import models, transforms

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def load_image(img_dir):
    img_list = []
    for filename in os.listdir(img_dir):
        img = Image.open(os.path.join(img_dir, filename))
        img = img.convert('RGB')
        transformed_img = transform(img)
        img_batch = torch.unsqueeze(transformed_img, 0)
        img_list.append(img_batch)
    return img_list


def initialize_model(weight_dir):
    model = ultralytics.YOLO(weight_dir)
    pytorch_model = model.model
    # the first run will fail, run the second time
    # input should be [batch_size, 3, width, height]
    wrapped_model = TensorRTWrapper(pytorch_model, [1, 3, 224, 224])
    return model, wrapped_model

# save the model as .engine, so that there is not need to set up 
def save_engine(engine, file_name):
    TRT_LOGGER = trt.Logger()
    with open(file_name, "wb") as f:
        f.write(engine.serialize())

def load_engine(engine_file_path):
    # load the saved engine
    trt_logger = trt.Logger()
    engine = trt.init_libnvinfer_plugins(trt_logger, "")
    print("Loading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f:
        engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
    
    return engine



def test_model(model, wrapped_model, img_list):
    import time
    # Time compilation with TensorRT
    start_time = time.time()
    for img in img_list:
        wrapped_model.predict(img)
    end_time = time.time()
    tensorrt_compile_time = end_time - start_time
    print(f"TensorRT compilation time: {tensorrt_compile_time:.4f} seconds")

    # Time compilation without TensorRT
    i = 1
    start_time = time.time()
    for img in  os.listdir(imgdir):
        # print(img)
        pred = model.predict(imgdir+img)
        print(pred)
        print(i)
        i+=1
    end_time = time.time()
    pytorch_compile_time = end_time - start_time

    print(f"PyTorch compilation time: {pytorch_compile_time:.4f} seconds")


imgdir = "/content/drive/MyDrive/Inference_withTorchTensorRT/CELL_imgs/"
load_image(imgdir)

weight_dir = "/content/drive/MyDrive/Inference_withTorchTensorRT/24_10_23_yolov8x_no_aug_iou_0.7.pt"
model, wrapped_model = initialize_model(weight_dir)

test_model(model, wrapped_model,imgdir)

engine_file_path = os.path.join("/content/drive/MyDrive/Inference_withTorchTensorRT", "yolov8x.engine")
save_engine(wrapped_model.model.engine, engine_file_path)

load_engine(engine_file_path)