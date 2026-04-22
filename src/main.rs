use clap::Parser;

use p2o::cli::Cli;
use p2o::converter::Converter;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    log::info!("=== PaddlePaddle to ONNX Converter ===");
    log::info!("Target Opset: {}", cli.opset);

    let mut converter = Converter::new();
    converter.set_target_opset(cli.opset);
    converter.set_strict(cli.strict);
    converter.load_paddle_model(&cli.model_json)?;
    converter.load_paddle_weights(&cli.model_pdiparams)?;
    converter.export_onnx(&cli.output_onnx, cli.opset)?;
    Ok(())
}
