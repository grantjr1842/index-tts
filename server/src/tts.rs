use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;
use std::sync::Arc;

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

        #[allow(deprecated)]
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

            // Optimization flags - configurable via environment variables (default: enabled)
            let use_fp16 = std::env::var("TARS_FP16")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true);
            let use_torch_compile = std::env::var("TARS_TORCH_COMPILE")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true);
            let use_accel = std::env::var("TARS_ACCEL")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(true);

            kwargs.set_item("use_fp16", use_fp16)?;
            kwargs.set_item("use_torch_compile", use_torch_compile)?;
            kwargs.set_item("use_accel", use_accel)?;

            println!("Model config: use_fp16={}, use_torch_compile={}, use_accel={}", 
                     use_fp16, use_torch_compile, use_accel);

            let model = cls.call((), Some(&kwargs))?;
            println!("Model loaded successfully!");
            Ok::<Py<PyAny>, PyErr>(model.into())
        })?;

        Ok(Self {
            model: Arc::new(model),
        })
    }

    pub fn infer(&self, params: TtsParams) -> PyResult<Vec<u8>> {
        let total_start = std::time::Instant::now();
        let text_len = params.text.len();
        
        #[allow(deprecated)]
        let result = Python::with_gil(|py| {
            let model = self.model.bind(py);
            let kwargs = PyDict::new(py);

            if !Path::new(&params.ref_audio).exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("Reference audio not found: {}", params.ref_audio),
                ));
            }

            kwargs.set_item("text", params.text)?;
            kwargs.set_item("spk_audio_prompt", params.ref_audio)?;
            kwargs.set_item("output_path", py.None())?;
            kwargs.set_item("temperature", params.temperature)?;
            kwargs.set_item("top_p", params.top_p)?;
            kwargs.set_item("top_k", params.top_k)?;
            kwargs.set_item(
                "max_text_tokens_per_segment",
                params.max_text_tokens_per_segment,
            )?;
            kwargs.set_item("return_audio", true)?;
            kwargs.set_item("return_numpy", true)?;
            kwargs.set_item("verbose", true)?;  // Enable Python-side timing logs

            let infer_start = std::time::Instant::now();
            let result = model.call_method("infer", (), Some(&kwargs))?;
            let infer_duration = infer_start.elapsed();
            
            // Get RTF from Python result if available
            let rtf = result.getattr("rtf")
                .ok()
                .and_then(|r| r.extract::<f64>().ok())
                .unwrap_or(0.0);
            let duration_sec = result.getattr("duration_sec")
                .ok()
                .and_then(|r| r.extract::<f64>().ok())
                .unwrap_or(0.0);
            
            println!("[TIMING] Python infer: {:?}, RTF: {:.4}, audio_duration: {:.2}s", 
                     infer_duration, rtf, duration_sec);

            // result is InferenceResult. audio is numpy array.
            let audio_numpy = result.getattr("audio")?;
            let sr = result.getattr("sampling_rate")?.extract::<i32>()?;

            let encode_start = std::time::Instant::now();
            let sf = py.import("soundfile")?;
            let io = py.import("io")?;
            let buffer = io.call_method0("BytesIO")?;

            sf.call_method1("write", (&buffer, audio_numpy, sr, "WAV"))?;
            let bytes = buffer.call_method0("getvalue")?.extract::<Vec<u8>>()?;
            let encode_duration = encode_start.elapsed();
            
            println!("[TIMING] WAV encoding: {:?}, bytes: {}", encode_duration, bytes.len());

            Ok(bytes)
        });
        
        println!("[TIMING] Total infer (text_len={}): {:?}", text_len, total_start.elapsed());
        result
    }

    pub fn infer_stream(&self, params: TtsParams) -> PyResult<Py<PyAny>> {
        #[allow(deprecated)]
        Python::with_gil(|py| {
            let model = self.model.bind(py);
            let kwargs = PyDict::new(py);

            if !Path::new(&params.ref_audio).exists() {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
                    format!("Reference audio not found: {}", params.ref_audio),
                ));
            }

            kwargs.set_item("text", params.text)?;
            kwargs.set_item("spk_audio_prompt", params.ref_audio)?;
            kwargs.set_item("output_path", py.None())?;
            kwargs.set_item("temperature", params.temperature)?;
            kwargs.set_item("top_p", params.top_p)?;
            kwargs.set_item("top_k", params.top_k)?;
            kwargs.set_item(
                "max_text_tokens_per_segment",
                params.max_text_tokens_per_segment,
            )?;
            kwargs.set_item("stream_return", true)?;

            let generator = model.call_method("infer", (), Some(&kwargs))?;

            // Return the generator object to be iterated in Rust
            Ok(generator.unbind())
        })
    }
}
