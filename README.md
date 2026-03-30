# Netproject1

计算机网络课程项目1源码工程。

本项目主链路采用 **WRGB 彩色二维码协议**，实现流程为：

- 输入二进制文件
- 编码为 WRGB 协议视频
- 手机拍摄播放中的视频
- 对拍摄视频逐帧矫正并解码
- 输出恢复得到的二进制文件和有效位图文件

目前采用协议为 144 x 144 的5色（数据区4色）二维码，一帧图片带有 34936 bit 数据量，30fps下理论传输速率约为 127.94 KiB/s 。

## 环境要求

- Windows
- Python 3.10 及以上
- `best.pt` 模型文件放在工程根目录

安装依赖：

```powershell
pip install -r requirements.txt
```

## 工程结构

```text
src/
  encode.py              编码主入口
  decode.py              解码主入口
  _color2Dcode.py        WRGB 协议门面
  color_codec.py         WRGB 编解码核心逻辑
  config.py              协议与模型配置
  utils/
    rectify.py           几何矫正与中心锚点修正
    rectify_tool.py      YOLO 矫正控制逻辑
    video_decode.py      视频拆帧与预处理
best.pt                  YOLO 模型权重
requirements.txt         运行依赖
```

## 编码

命令格式：

```powershell
python src\encode.py <input_file> <output_video> <max_length_ms> [fps]
```

参数说明：

- `input_file`：待编码二进制文件
- `output_video`：输出视频路径
- `max_length_ms`：视频最大时长，单位毫秒
- `fps`：可选帧率，缺省为 `30`

示例：

```powershell
python src\encode.py input.bin out.mp4 1000
python src\encode.py input.bin out.mp4 1000 15
```

## 解码

命令格式：

```powershell
python src\decode.py <input_video> <output_bin> <output_vbin>
```

参数说明：

- `input_video`：拍摄得到的视频文件
- `output_bin`：恢复出的二进制文件
- `output_vbin`：有效位图文件

示例：

```powershell
python src\decode.py phone.mp4 out.bin vout.bin
```

如需导出调试信息：

```powershell
python src\decode.py phone.mp4 out.bin vout.bin --debug --debug-dir output\rectify_debug
```
## 说明

- 项目测试时，生成的视频为 1440x1440 30fps ，手机拍摄规格为4k60fps。
- 手机要拿稳或者使用手机支架等工具。
- 视频时长超出限制时将会进行截断，最终解码出的文件将短于源文件
