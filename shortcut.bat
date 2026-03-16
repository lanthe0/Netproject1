@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo 开始编码
python src/encode.py data/test_random.bin output/test.mp4 15000
timeout /t 5 /nobreak >nul
echo 编码完成，开始解码
python src/decode.py output/test.mp4 output/decode.bin output/vout.bin
echo 解码成功
pause