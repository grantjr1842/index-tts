use axum::{
    extract::{Json, State},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use std::net::SocketAddr;

mod tts;
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
