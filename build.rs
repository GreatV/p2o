use serde::Deserialize;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct CodegenConfig {
    pass2_dispatch: Vec<DispatchEntry>,
    op_type_map: Vec<OpTypeEntry>,
}

#[derive(Debug, Deserialize)]
struct DispatchEntry {
    op: String,
    kind: DispatchKind,
    method: String,
    args: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DispatchKind {
    Noop,
    Simple,
    Typed,
    Arg,
}

#[derive(Debug, Deserialize)]
struct OpTypeEntry {
    paddle: Vec<String>,
    onnx: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=proto/framework.proto");
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    println!("cargo:rerun-if-changed=codegen/ops.yaml");

    prost_build::compile_protos(&["proto/framework.proto", "proto/onnx.proto"], &["proto/"])?;

    let config: CodegenConfig = serde_yaml::from_str(&fs::read_to_string("codegen/ops.yaml")?)?;
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    fs::write(
        out_dir.join("pass2_dispatch.rs"),
        generate_pass2_dispatch(&config.pass2_dispatch),
    )?;
    fs::write(
        out_dir.join("op_type_lookup.rs"),
        generate_op_type_lookup(&config.op_type_map),
    )?;

    Ok(())
}

fn generate_pass2_dispatch(entries: &[DispatchEntry]) -> String {
    let mut output = String::from(
        "fn build_pass2_op_dispatch() -> HashMap<&'static str, Pass2OpHandler> {\n    HashMap::from([\n",
    );
    for entry in entries {
        let line = match entry.kind {
            DispatchKind::Noop => {
                format!(
                    "        (\"{}\", pass2_noop as Pass2OpHandler),\n",
                    entry.op
                )
            }
            DispatchKind::Simple => {
                format!(
                    "        simple_dispatch!(\"{}\" => {}),\n",
                    entry.op, entry.method
                )
            }
            DispatchKind::Typed => {
                format!(
                    "        typed_dispatch!(\"{}\" => {}),\n",
                    entry.op, entry.method
                )
            }
            DispatchKind::Arg => {
                let args = entry.args.as_deref().unwrap_or(&[]).join(", ");
                format!(
                    "        arg_dispatch!(\"{}\" => {}({})),\n",
                    entry.op, entry.method, args
                )
            }
        };
        output.push_str(&line);
    }
    output.push_str("    ])\n}\n");
    output
}

fn generate_op_type_lookup(entries: &[OpTypeEntry]) -> String {
    let mut output = String::from(
        "fn generated_op_type(paddle_op: &str) -> Option<&'static str> {\n    match paddle_op {\n",
    );
    for entry in entries {
        let patterns = entry
            .paddle
            .iter()
            .map(|name| format!("\"{}\"", name))
            .collect::<Vec<_>>()
            .join(" | ");
        output.push_str(&format!(
            "        {} => Some(\"{}\"),\n",
            patterns, entry.onnx
        ));
    }
    output.push_str("        _ => None,\n    }\n}\n");
    output
}
