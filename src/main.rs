mod cli;
mod proto;
mod helper;
mod converter;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = cli::Cli::parse();
    log::info!("=== PaddlePaddle to ONNX Converter ===");
    log::info!("Target Opset: {}", cli.opset);

    let mut converter = converter::Converter::new();
    converter.load_paddle_model(&cli.model_json)?;
    converter.load_paddle_weights(&cli.model_pdiparams)?;
    converter.export_onnx(&cli.output_onnx, cli.opset)?;
    Ok(())
}
