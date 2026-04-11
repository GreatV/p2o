use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(&["proto/framework.proto", "proto/onnx.proto"], &["proto/"])?;
    Ok(())
}
