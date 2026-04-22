#[allow(clippy::all)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[allow(dead_code)]
#[allow(clippy::all)]
pub mod paddle {
    pub mod framework {
        pub mod proto {
            pub mod var_type {
                include!(concat!(env!("OUT_DIR"), "/paddle.framework.proto.rs"));
            }
        }
    }
}
// prost preserves Paddle's nested VarType message as var_type::var_type.
pub use paddle::framework::proto::var_type::var_type::TensorDesc;
pub use paddle::framework::proto::var_type::var_type::Type as PaddleDataType;
