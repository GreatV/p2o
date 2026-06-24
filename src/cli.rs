use clap::{Parser, ValueEnum};

use crate::converter::{DEFAULT_OPSET, OptimizationLevel};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetOpset {
    Auto,
    Version(i64),
}

impl TargetOpset {
    pub fn resolve(&self) -> i64 {
        match self {
            Self::Auto => DEFAULT_OPSET,
            Self::Version(opset) => *opset,
        }
    }
}

fn parse_target_opset(raw: &str) -> Result<TargetOpset, String> {
    if raw.eq_ignore_ascii_case("auto") {
        return Ok(TargetOpset::Auto);
    }
    let opset = raw
        .parse::<i64>()
        .map_err(|_| format!("Invalid opset value: {raw}"))?;
    if opset < 10 {
        return Err(format!(
            "Target ONNX opset must be >= 10, got {opset}. Lower opsets are not supported."
        ));
    }
    Ok(TargetOpset::Version(opset))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum CliOptimizationLevel {
    None,
    Basic,
}

impl From<CliOptimizationLevel> for OptimizationLevel {
    fn from(value: CliOptimizationLevel) -> Self {
        match value {
            CliOptimizationLevel::None => OptimizationLevel::None,
            CliOptimizationLevel::Basic => OptimizationLevel::Basic,
        }
    }
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
        default_value = "17",
        value_parser = parse_target_opset,
        help = "Target ONNX opset version (>= 10, e.g. 17) or auto"
    )]
    pub opset: TargetOpset,

    /// ONNX graph optimization level.
    #[arg(long, value_enum, default_value_t = CliOptimizationLevel::Basic)]
    pub optimize: CliOptimizationLevel,

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
        assert_eq!(cli.opset.resolve(), crate::converter::DEFAULT_OPSET);
        assert_eq!(cli.optimize, super::CliOptimizationLevel::Basic);
    }

    #[test]
    fn test_cli_accepts_auto_opset() {
        let cli = Cli::try_parse_from([
            "p2o",
            "model.json",
            "model.pdiparams",
            "model.onnx",
            "--opset",
            "auto",
        ])
        .unwrap();
        assert_eq!(cli.opset, super::TargetOpset::Auto);
        assert_eq!(cli.opset.resolve(), crate::converter::DEFAULT_OPSET);
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
