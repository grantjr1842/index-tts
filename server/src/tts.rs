use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use std::path::Path;

#[derive(Clone)]
pub struct TtsModel {
    model: Arc<Py<PyAny>>,
}

#[derive(Debug, Clone)]
pub struct TtsParams {
    pub text: String,
    pub ref_audio: String,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub max_text_tokens_per_segment: i32,
}

impl TtsModel {
    pub fn new() -> PyResult<Self> {
        println!("Initializing Python...");
        // pyo3::prepare_freethreaded_python(); // Already called or handled by auto-initialize

        let model = Python::with_gil(|py| {
            let sys = py.import("sys")?;
            // Add current directory to sys.path so we can find indextts
            sys.getattr("path")?.call_method1("append", (".",))?;
            
            println!("Loading IndexTTS2...");
            let indextts = py.import("indextts.infer_v2")?;
            let cls = indextts.getattr("IndexTTS2")?;
            
            let kwargs = PyDict::new(py);
            kwargs.set_item("cfg_path", "checkpoints/config.yaml")?;
            kwargs.set_item("model_dir", "checkpoints")?;
            
            // Use environment variable for device or default to cuda:0 if available
            let device = std::env::var("TARS_DEVICE").unwrap_or_else(|_| "cuda:0".to_string());
            kwargs.set_item("device", device)?;
            
            // Match serve_tars.py defaults
            kwargs.set_item("use_fp16", true)?;
            
            let model = cls.call((), Some(&kwargs))?;
            println!("Model loaded successfully!");
            Ok::<Py<PyAny>, PyErr>(model.into())
        })?;

        Ok(Self {
            model: Arc::new(model),
        })
    }

    pub fn infer(&self, params: TtsParams) -> PyResult<Vec<u8>> {
        Python::with_gil(|py| {
            let model = self.model.bind(py);
            let kwargs = PyDict::new(py);
            
            if !Path::new(&params.ref_audio).exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("Reference audio not found: {}", params.ref_audio)
                ));
            }

            kwargs.set_item("text", params.text)?;
            kwargs.set_item("spk_audio_prompt", params.ref_audio)?;
            kwargs.set_item("output_path", py.None())?;
            kwargs.set_item("temperature", params.temperature)?;
            kwargs.set_item("top_p", params.top_p)?;
            kwargs.set_item("top_k", params.top_k)?;
            kwargs.set_item("max_text_tokens_per_segment", params.max_text_tokens_per_segment)?;
            kwargs.set_item("return_audio", true)?;
            kwargs.set_item("return_numpy", true)?;

            let result = model.call_method("infer", (), Some(&kwargs))?;
            
            // result is InferenceResult. audio is numpy array.
            let audio_numpy = result.getattr("audio")?;
            let sr = result.getattr("sampling_rate")?.extract::<i32>()?;
            
            let sf = py.import("soundfile")?;
            let io = py.import("io")?;
            let buffer = io.call_method0("BytesIO")?;
            
            sf.call_method1("write", (&buffer, audio_numpy, sr, "WAV"))?;
            let bytes = buffer.call_method0("getvalue")?.extract::<Vec<u8>>()?;
            
            Ok(bytes)
        })
    }

    pub fn infer_stream(&self, params: TtsParams) -> PyResult<Py<PyAny>> {
        Python::with_gil(|py| {
            let model = self.model.bind(py);
            let kwargs = PyDict::new(py);
            
            if !Path::new(&params.ref_audio).exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("Reference audio not found: {}", params.ref_audio)
                ));
            }

            kwargs.set_item("text", params.text)?;
            kwargs.set_item("spk_audio_prompt", params.ref_audio)?;
            kwargs.set_item("output_path", py.None())?;
            kwargs.set_item("temperature", params.temperature)?;
            kwargs.set_item("top_p", params.top_p)?;
            kwargs.set_item("top_k", params.top_k)?;
            kwargs.set_item("max_text_tokens_per_segment", params.max_text_tokens_per_segment)?;
            kwargs.set_item("stream_return", true)?;

            let generator = model.call_method("infer", (), Some(&kwargs))?;
            
            // Return the generator object to be iterated in Rust
            Ok(generator.unbind())
        })
    }
}
