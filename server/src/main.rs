use axum::body::{Body, Bytes};
use axum::{
    Router,
    extract::{Json, State},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use std::net::SocketAddr;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

mod tts;
use pyo3::prelude::*;
use pyo3::types::PyIterator;
use tts::{TtsModel, TtsParams};

#[derive(Clone)]
struct AppState {
    tts_model: TtsModel,
}

#[derive(Deserialize)]
struct TTSRequest {
    text: String,
    #[serde(default = "default_temp")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default = "default_top_k")]
    top_k: i32,
    #[serde(default = "default_speed")]
    _speed: f32,
    #[serde(default = "default_tokens")]
    max_text_tokens_per_segment: i32,
}

fn default_temp() -> f32 {
    0.8
}
fn default_top_p() -> f32 {
    0.8
}
fn default_top_k() -> i32 {
    30
}
fn default_speed() -> f32 {
    1.0
}
fn default_tokens() -> i32 {
    120
}

// ... imports ...
use anyhow::Context;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // pyo3::prepare_freethreaded_python(); // Deprecated and handled by auto-initialize

    let tts_model = TtsModel::new().expect("Failed to load TTS model");

    // Warmup: Perform a dummy inference to cache speaker embeddings and trigger torch.compile
    let warmup_enabled = std::env::var("TARS_WARMUP")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true);
    
    if warmup_enabled {
        println!("Starting warmup inference...");
        let ref_audio = std::env::var("TARS_REFERENCE_AUDIO")
            .unwrap_or_else(|_| "interstellar-tars-01-resemble-denoised.wav".to_string());
        
        let warmup_params = TtsParams {
            // Use minimal text to reduce memory for warmup
            text: "Hi".to_string(),
            ref_audio,
            temperature: 0.8,
            top_p: 0.8,
            top_k: 30,
            max_text_tokens_per_segment: 120,
        };
        
        let warmup_start = std::time::Instant::now();
        match tts_model.infer(warmup_params) {
            Ok(_) => {
                println!("Warmup completed in {:?}", warmup_start.elapsed());
            }
            Err(e) => {
                eprintln!("Warmup inference failed (non-fatal): {}", e);
                
                // Print GPU memory diagnostics on failure
                #[allow(deprecated)]
                let _ = Python::with_gil(|py| {
                    if let Ok(torch) = py.import("torch") {
                        if let Ok(cuda) = torch.getattr("cuda") {
                            if let Ok(true) = cuda.getattr("is_available")?.call0()?.extract::<bool>() {
                                if let Ok(allocated) = cuda.call_method0("memory_allocated") {
                                    if let Ok(reserved) = cuda.call_method0("memory_reserved") {
                                        if let Ok(total) = cuda.getattr("get_device_properties")?.call1((0,))?.getattr("total_memory") {
                                            eprintln!("GPU Memory - Allocated: {} bytes, Reserved: {} bytes, Total: {} bytes", 
                                                allocated, reserved, total);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Ok::<(), PyErr>(())
                });
            }
        }
    } else {
        println!("Warmup disabled via TARS_WARMUP=0");
    }

    let state = AppState { tts_model };

    let app = Router::new()
        .route("/healthz", get(health_check))
        .route("/tts", post(tts_handler))
        .route("/tts/stream", post(stream_handler))
        .with_state(state);

    let port = 8009;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("Failed to bind to address")?;
    axum::serve(listener, app).await.context("Server error")?;

    Ok(())
}

async fn health_check() -> &'static str {
    "ok"
}

async fn stream_handler(
    State(state): State<AppState>,
    Json(payload): Json<TTSRequest>,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);

    let stream = ReceiverStream::new(rx);
    let body = Body::from_stream(stream);

    tokio::task::spawn_blocking(move || {
        let ref_audio = std::env::var("TARS_REFERENCE_AUDIO")
            .unwrap_or_else(|_| "interstellar-tars-01-resemble-denoised.wav".to_string());

        let params = TtsParams {
            text: payload.text,
            ref_audio,
            temperature: payload.temperature,
            top_p: payload.top_p,
            top_k: payload.top_k,
            max_text_tokens_per_segment: payload.max_text_tokens_per_segment,
        };

        let result = state.tts_model.infer_stream(params);

        match result {
            Ok(generator) => {
                let sender = tx;

                #[allow(deprecated)]
                let iter_result = Python::with_gil(|py| {
                    let iter_gen = generator.bind(py);
                    let iter = PyIterator::from_object(iter_gen)?;

                    // Optimization: Import modules once outside the loop
                    let sf = py.import("soundfile")?;
                    let io = py.import("io")?;

                    for item in iter {
                        let chunk = item?;

                        let wav_bytes: Vec<u8> = (|| -> PyResult<Vec<u8>> {
                            let chunk_tensor = chunk;
                            let chunk_tensor = chunk_tensor.call_method1("squeeze", (0,))?;
                            let chunk_cpu = chunk_tensor.call_method0("cpu")?;
                            let wav_numpy = chunk_cpu.call_method0("numpy")?;

                            let buffer = io.call_method0("BytesIO")?;

                            // Use keyword argument for format when writing to BytesIO
                            let sf_kwargs = pyo3::types::PyDict::new(py);
                            sf_kwargs.set_item("format", "WAV")?;
                            sf.call_method("write", (&buffer, wav_numpy, 22050), Some(&sf_kwargs))?;
                            buffer.call_method0("getvalue")?.extract::<Vec<u8>>()
                        })()?;

                        if sender.blocking_send(Ok(Bytes::from(wav_bytes))).is_err() {
                            break;
                        }
                    }
                    Ok::<(), PyErr>(())
                });

                if let Err(e) = iter_result {
                    eprintln!("Streaming error: {}", e);
                }
            }
            Err(e) => {
                eprintln!("Failed to start stream: {}", e);
            }
        }
    });

    Response::builder()
        .header("Content-Type", "audio/wav")
        .header("Transfer-Encoding", "chunked")
        .body(body)
        .expect("Failed to build streaming response")
}

async fn tts_handler(State(state): State<AppState>, Json(payload): Json<TTSRequest>) -> Response {
    let result = tokio::task::spawn_blocking(move || {
        let ref_audio = std::env::var("TARS_REFERENCE_AUDIO")
            .unwrap_or_else(|_| "interstellar-tars-01-resemble-denoised.wav".to_string());

        let params = TtsParams {
            text: payload.text,
            ref_audio,
            temperature: payload.temperature,
            top_p: payload.top_p,
            top_k: payload.top_k,
            max_text_tokens_per_segment: payload.max_text_tokens_per_segment,
        };

        state.tts_model.infer(params)
    })
    .await;

    // Handle potential task panic
    let result = match result {
        Ok(r) => r,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Task panicked: {}", e),
            )
                .into_response();
        }
    };

    match result {
        Ok(bytes) => ([(axum::http::header::CONTENT_TYPE, "audio/wav")], bytes).into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error: {}", e),
        )
            .into_response(),
    }
}
