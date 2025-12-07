use axum::{
    extract::{Json, State},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use axum::body::{Body, Bytes};
use serde::Deserialize;
use std::net::SocketAddr;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

mod tts;
use tts::{TtsModel, TtsParams};
use pyo3::prelude::*;
use pyo3::types::PyIterator;

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
    speed: f32,
    #[serde(default = "default_tokens")]
    max_text_tokens_per_segment: i32,
}

fn default_temp() -> f32 { 0.8 }
fn default_top_p() -> f32 { 0.8 }
fn default_top_k() -> i32 { 30 }
fn default_speed() -> f32 { 1.0 }
fn default_tokens() -> i32 { 120 }

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    pyo3::prepare_freethreaded_python();
    
    let tts_model = TtsModel::new().expect("Failed to load TTS model");

    let state = AppState {
        tts_model,
    };

    let app = Router::new()
        .route("/healthz", get(health_check))
        .route("/tts", post(tts_handler))
        .route("/tts/stream", post(stream_handler))
        .with_state(state);

    let port = 8009;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
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
                
                let iter_result = Python::with_gil(|py| {
                    let iter_gen = generator.bind(py);
                    let iter = PyIterator::from_object(iter_gen)?;
                    
                    for item in iter {
                        let chunk = item?;
                        
                        let wav_bytes: Vec<u8> = (|| -> PyResult<Vec<u8>> {
                            let chunk_tensor = chunk;
                            let chunk_tensor = chunk_tensor.call_method1("squeeze", (0,))?;
                            let chunk_cpu = chunk_tensor.call_method0("cpu")?;
                            let wav_numpy = chunk_cpu.call_method0("numpy")?;
                            
                            let sf = py.import("soundfile")?;
                            let io = py.import("io")?;
                            let buffer = io.call_method0("BytesIO")?;
                            
                            sf.call_method1("write", (&buffer, wav_numpy, 24000, "WAV"))?;
                            buffer.call_method0("getvalue")?.extract::<Vec<u8>>()
                        })()?;
                        
                        if let Err(_) = sender.blocking_send(Ok(Bytes::from(wav_bytes))) {
                             break;
                        }
                    }
                    Ok::<(), PyErr>(())
                });
                
                if let Err(e) = iter_result {
                    eprintln!("Streaming error: {}", e);
                }
            },
            Err(e) => {
                eprintln!("Failed to start stream: {}", e);
            }
        }
    });

    Response::builder()
        .header("Content-Type", "audio/wav")
        .header("Transfer-Encoding", "chunked")
        .body(body)
        .unwrap()
}

async fn tts_handler(
    State(state): State<AppState>,
    Json(payload): Json<TTSRequest>,
) -> Response {
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
    }).await.unwrap();

    match result {
        Ok(bytes) => {
             (
                 [(axum::http::header::CONTENT_TYPE, "audio/wav")],
                 bytes
             ).into_response()
        },
        Err(e) => {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Error: {}", e)
            ).into_response()
        }
    }
}
