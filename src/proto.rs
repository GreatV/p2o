pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[allow(dead_code)]
pub mod paddle {
    pub mod framework {
        pub mod proto {
            pub mod var_type {
                include!(concat!(env!("OUT_DIR"), "/paddle.framework.proto.rs"));
            }
        }
    }
}
pub use paddle::framework::proto::var_type::var_type::TensorDesc;
