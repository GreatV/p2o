use clap::Parser;

use p2o::cli::Cli;
use p2o::converter::Converter;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    let opset = cli.opset.resolve();
    log::info!("=== PaddlePaddle to ONNX Converter ===");
    log::info!("Target Opset: {}", opset);
    log::info!("Optimization Level: {:?}", cli.optimize);

    let mut converter = Converter::new();
    converter.set_target_opset(opset);
    converter.set_optimization_level(cli.optimize.into());
    converter.set_strict(cli.strict);
    converter.load_paddle_model(&cli.model_json)?;
    converter.load_paddle_weights(&cli.model_pdiparams)?;
    converter.export_onnx(&cli.output_onnx, opset)?;
    Ok(())
}
