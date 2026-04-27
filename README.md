# p2o

A PaddlePaddle New IR (PIR) to ONNX model converter, written in Rust.

Converts PaddlePaddle inference models (`inference.json` + `inference.pdiparams`) to the ONNX format.

## Installation

### Pre-built binaries

Download from [GitHub Releases](https://github.com/greatv/p2o/releases).

### From PyPI

```bash
pip install p2o
```

### From crates.io

```bash
cargo install p2o
```

### Build from source

```bash
git clone https://github.com/greatv/p2o.git
cd p2o
cargo build --release
```

## Usage

```bash
p2o <model.json> <model.pdiparams> <output.onnx> [--opset 17] [--strict]
```

### Arguments

| Argument | Description |
|---|---|
| `model.json` | Path to the PaddlePaddle `.json` model file |
| `model.pdiparams` | Path to the PaddlePaddle `.pdiparams` weight file |
| `output.onnx` | Path to the output `.onnx` model file |
| `--opset <N>` | Target ONNX opset version (≥ 10, default: 17) |
| `--strict` | Reject lossy conversions (e.g. multinomial → ArgMax) |

### Example

```bash
p2o inference_models/PP-OCRv5_server_det_infer/inference.json \
    inference_models/PP-OCRv5_server_det_infer/inference.pdiparams \
    output.onnx --opset 17
```
