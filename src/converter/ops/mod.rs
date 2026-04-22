use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

mod array;
mod control;
mod conv;
mod indexing;
mod math;
mod misc;
mod nms;
mod norm;
mod pad;
mod reduce;
mod resize;
mod shape;
mod transform;
mod update;

type Pass2OpHandler = fn(&mut super::Converter, &str, &Value) -> anyhow::Result<()>;

static PASS2_OP_DISPATCH: OnceLock<HashMap<&'static str, Pass2OpHandler>> = OnceLock::new();

fn pass2_noop(_: &mut super::Converter, _: &str, _: &Value) -> anyhow::Result<()> {
    Ok(())
}

macro_rules! simple_dispatch {
    ($op:expr => $method:ident) => {
        (
            $op,
            (|this: &mut super::Converter, _: &str, op: &Value| this.$method(op)) as Pass2OpHandler,
        )
    };
}

macro_rules! typed_dispatch {
    ($op:expr => $method:ident) => {
        (
            $op,
            (|this: &mut super::Converter, op_type: &str, op: &Value| this.$method(op_type, op))
                as Pass2OpHandler,
        )
    };
}

macro_rules! arg_dispatch {
    ($op:expr => $method:ident($($arg:expr),+ $(,)?)) => {
        (
            $op,
            (|this: &mut super::Converter, _: &str, op: &Value| this.$method(op, $($arg),+))
                as Pass2OpHandler,
        )
    };
}

include!(concat!(env!("OUT_DIR"), "/pass2_dispatch.rs"));

fn pass2_op_dispatch() -> &'static HashMap<&'static str, Pass2OpHandler> {
    PASS2_OP_DISPATCH.get_or_init(build_pass2_op_dispatch)
}

impl super::Converter {
    pub fn process_pass2_op(&mut self, op_type: &str, op: &Value) -> anyhow::Result<()> {
        if let Some(handler) = pass2_op_dispatch().get(op_type) {
            return handler(self, op_type, op);
        }

        if op_type.starts_with("1.") {
            log::debug!(
                "No dedicated handler for {}; using generic conversion",
                op_type
            );
            // Convention: Paddle trailing '_' denotes an inplace variant with the same
            // operator semantics for conversion purposes, so `relu_` reuses `relu`.
            let clean_op_type = op_type.trim_start_matches("1.").trim_end_matches("_");
            self.convert_generic_op(clean_op_type, op)?;
        }
        Ok(())
    }
}
