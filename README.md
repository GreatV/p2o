# p2o (PaddlePaddle to ONNX Converter)

A fast, Rust-based converter from PaddlePaddle's new static graph format (PIR JSON + `.pdiparams`) to ONNX.
This tool supports transforming PP-OCRv5 detection and recognition models into ONNX format.

## Usage

```bash
# Build the converter
cargo build --release

# Run conversion
cargo run --release -- <model.json> <model.pdiparams> <output.onnx> --opset 15
```


## Supported Operators

Currently, the converter natively handles many operators used in the PP-OCRv5 pipeline, including:
- Conv2d / DepthwiseConv2d / Conv2dTranspose
- BatchNorm / LayerNorm
- MaxPool / AveragePool
- MatMul / Add / Mul / Reshape / Transpose / Slice / Stack / Squeeze / Unsqueeze
- Swish / Sigmoid / HardSwish
- Nearest Interp (Resize)
- ... and more.
