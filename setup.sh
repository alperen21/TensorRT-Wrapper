pip install onnx
pip install onnxruntime
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
# ensure torch and torchvision has the same versioin
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
pip install torch_tensorrt
cd torch2trt
python setup.py install
cd ..
mv torch2trt torch2trt_github
cp -r torch2trt_github/torch2trt torch2trt
pip install pycuda
pip install torch_tensorrt

