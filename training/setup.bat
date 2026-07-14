@echo off
echo Setting up Python environment...
python -m venv training\venv
call training\venv\Scripts\activate
python -m ensurepip --upgrade

echo.
echo Installing PyTorch and dependencies...
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime numpy pymupdf pillow torchvision
if exist training\requirements.txt (
  pip install -r training\requirements.txt
)
if exist training\trocr\requirements.txt (
  pip install -r training\trocr\requirements.txt
)

echo.
echo Verifying installation...
python -c "import fitz, PIL, torch, onnxruntime; print('Post-install check passed: all required modules imported successfully.')"
if %ERRORLEVEL% NEQ 0 (
  echo [ERROR] Post-install check failed!
  exit /b 1
)

echo.
echo Done.
echo.
echo To train:
echo   training\venv\Scripts\activate
echo   python training\train.py data\jv.txt
