@echo off
echo Setting up Python environment...
python -m venv training\venv
call training\venv\Scripts\activate

echo.
echo Installing PyTorch and dependencies...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime numpy

echo.
echo Done.
echo.
echo To train:
echo   training\venv\Scripts\activate
echo   python training\train.py data\jv.txt
