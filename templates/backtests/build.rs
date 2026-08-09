//! Records the interpreter the strategy embeds at run time (see
//! `src/interpreter.rs`).
//!
//! Preference order: the repository's uv virtual environment, whose
//! `site-packages` carry the operators' dependencies (numpy, scipy, cvxpy),
//! then the interpreter `pyo3-build-config` resolved and linked against —
//! both share the same base installation, so the choice only decides which
//! environment's packages the embedded interpreter sees.

use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo::rerun-if-env-changed=VIRTUAL_ENV");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let venv = manifest.parent().unwrap().parent().unwrap().join(".venv");
    let executable = [venv.join("Scripts/python.exe"), venv.join("bin/python")]
        .into_iter()
        .find(|p| p.exists())
        .map(|p| p.to_string_lossy().into_owned())
        .or_else(|| pyo3_build_config::get().executable().map(str::to_owned));

    if let Some(executable) = executable {
        println!("cargo::rustc-env=STRATEGY_PYTHON_EXECUTABLE={executable}");
    }
}
