use clap::Parser;

use crate::converter::DEFAULT_OPSET;

fn parse_supported_opset(raw: &str) -> Result<i64, String> {
    let opset = raw
        .parse::<i64>()
        .map_err(|_| format!("Invalid opset value: {raw}"))?;
    if opset < 10 {
        return Err(format!(
            "Target ONNX opset must be >= 10, got {opset}. Lower opsets are not supported."
        ));
    }
    Ok(opset)
}

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
    #[arg(
        long,
        default_value_t = DEFAULT_OPSET,
        value_parser = parse_supported_opset,
        help = "Target ONNX opset version (>= 10, e.g. 17)"
    )]
    pub opset: i64,

    /// Fail instead of applying lossy compatibility lowerings.
    #[arg(long, help = "Reject lossy conversions such as multinomial -> ArgMax")]
    pub strict: bool,
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn test_cli_defaults_to_opset_17() {
        let cli =
            Cli::try_parse_from(["p2o", "model.json", "model.pdiparams", "model.onnx"]).unwrap();
        assert_eq!(cli.opset, crate::converter::DEFAULT_OPSET);
    }

    #[test]
    fn test_cli_rejects_unsupported_opset() {
        let err = Cli::try_parse_from([
            "p2o",
            "model.json",
            "model.pdiparams",
            "model.onnx",
            "--opset",
            "9",
        ])
        .unwrap_err();
        assert!(err.to_string().contains("must be >= 10"));
    }
}
