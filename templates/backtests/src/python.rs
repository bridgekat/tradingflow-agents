//! Startup of the embedded interpreter, as the repository's virtual
//! environment.
//!
//! TradingFlow leaves interpreter startup to the host: CPython's default
//! path computation resolves the base installation the linked `libpython`
//! belongs to, which would miss the uv environment's `site-packages` (numpy,
//! scipy, cvxpy). So `build.rs` records the environment's interpreter, and
//! [`initialize`] sets `PyConfig.executable` to it before startup — CPython
//! then behaves exactly as if that interpreter had been run directly, reading
//! its `pyvenv.cfg` and putting its `site-packages` on `sys.path`. No
//! `PYTHONHOME` or `PYTHONPATH` needed; both still override if set.

use std::sync::Once;

use libc::wchar_t;
use pyo3::ffi;
use pyo3::prelude::*;

/// Absolute path of the interpreter to embed, from `build.rs`.
///
/// `None` when the build could not name one, in which case startup is left to
/// CPython's own path computation.
const EXECUTABLE: Option<&str> = option_env!("STRATEGY_PYTHON_EXECUTABLE");

static START: Once = Once::new();

/// Starts the embedded interpreter, unless one is already running, and puts
/// the crate's `python/` directory on `sys.path` so `py_operator_module`
/// resolves the strategy-local operators like installed packages.
///
/// Idempotent, and cheap after the first call. Must run before the first
/// Python operator node is built — `pyo3::Python::attach` panics on an
/// uninitialized interpreter.
///
/// # Panics
///
/// Exits the process if CPython cannot start, the only thing a failed
/// interpreter startup permits — there is no interpreter left to raise with.
pub fn initialize(python_ops_dir: &str) {
    START.call_once(|| {
        // SAFETY: `Py_IsInitialized` is callable without an interpreter — that
        // is the question it answers.
        if unsafe { ffi::Py_IsInitialized() } == 0
            && let Some(executable) = EXECUTABLE
        {
            // SAFETY: no interpreter is running, as just checked, and `Once`
            // makes this the only thread that can be here.
            unsafe { initialize_as(executable) };
        }
        // Adopts what the block above built or, if `EXECUTABLE` was `None`,
        // performs the default startup.
        Python::initialize();

        let code =
            std::ffi::CString::new(format!("import sys; sys.path.append({:?})", python_ops_dir,))
                .unwrap();
        Python::attach(|py| py.run(&code, None, None).expect("cannot extend sys.path"));
    });
}

/// Starts CPython as though it had been launched as `executable`.
///
/// # Safety
///
/// No interpreter may be running, and no other thread may be starting one.
unsafe fn initialize_as(executable: &str) {
    unsafe {
        let mut config = std::mem::zeroed::<ffi::PyConfig>();
        ffi::PyConfig_InitPythonConfig(&mut config);

        // Everything else — prefix, standard library, `site-packages`, and
        // whether this is a virtual environment at all — CPython derives from
        // this one path, exactly as it does when the interpreter is launched
        // from disk.
        let path = wide(executable);
        let status = ffi::PyConfig_SetString(&mut config, &mut config.executable, path.as_ptr());
        if ffi::PyStatus_Exception(status) != 0 {
            ffi::PyConfig_Clear(&mut config);
            ffi::Py_ExitStatusException(status);
        }

        let status = ffi::Py_InitializeFromConfig(&config);
        ffi::PyConfig_Clear(&mut config);
        if ffi::PyStatus_Exception(status) != 0 {
            ffi::Py_ExitStatusException(status);
        }

        // Startup leaves this thread holding the GIL. Operator nodes run on
        // the engine's worker threads, so holding it here would deadlock the
        // first attach from any of them; PyO3 acquires it per attach.
        ffi::PyEval_SaveThread();
    }
}

/// Encodes a path as the NUL-terminated `wchar_t` string `PyConfig` wants.
fn wide(path: &str) -> Vec<wchar_t> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;

        std::ffi::OsStr::new(path)
            .encode_wide()
            .map(|unit| unit as wchar_t)
            .chain([0])
            .collect()
    }
    #[cfg(not(windows))]
    {
        // Elsewhere `wchar_t` holds a code point, which is what `chars` yields.
        path.chars().map(|c| c as wchar_t).chain([0]).collect()
    }
}
