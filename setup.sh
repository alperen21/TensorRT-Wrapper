pip install onnx
pip install onnxruntime
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
pip install tensorrt
cd torch2trt
python setup.py install
cd ..
mv torch2trt torch2trt_github
cp -r torch2trt_github/torch2trt torch2trt
pip install pycuda
pip install torch_tensorrt

