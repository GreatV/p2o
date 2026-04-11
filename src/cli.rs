use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "PaddlePaddle to ONNX Converter", long_about = None)]
pub struct Cli {
    /// PaddlePaddle model JSON file path
    #[arg(help = "Path to the PaddlePaddle .json model file")]
    pub model_json: String,

    /// PaddlePaddle model pdiparams file path
    #[arg(help = "Path to the PaddlePaddle .pdiparams weight file")]
    pub model_pdiparams: String,

    /// Output ONNX model file path
    #[arg(help = "Path to the output .onnx model file")]
    pub output_onnx: String,

    /// Target ONNX Opset version
    #[arg(long, default_value_t = 15, help = "Target ONNX opset version (e.g. 15)")]
    pub opset: i64,
}
