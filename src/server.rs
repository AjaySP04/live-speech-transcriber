use crate::config::Config;
use crate::db::Db;
use crate::pipeline::{run_pipeline, PipelineDeps, PipelineEvent};
use crate::whisper::WhisperEngine;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, Query, State,
    },
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{Arc, Mutex, RwLock};
use tower_http::services::ServeDir;

pub const AVAILABLE_MODELS: &[&str] = &["tiny", "base", "small", "medium", "large-v3"];

/// Settings the UI can change at runtime. Model/target changes apply to the
/// NEXT "Start listening" (each WS connection snapshots them at spawn).
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSettings {
    pub model: String,
    pub target_lang: String,
    pub allowed_languages: Vec<String>,
    pub loading: bool,
    pub error: Option<String>,
}

pub struct AppState {
    pub whisper: RwLock<Arc<WhisperEngine>>,
    pub db: Arc<Mutex<Db>>,
    pub cfg: Config,
    pub settings: RwLock<RuntimeSettings>,
}

impl AppState {
    pub fn runtime_settings(cfg: &Config) -> RuntimeSettings {
        RuntimeSettings {
            model: cfg.whisper_model.clone(),
            target_lang: cfg.target_lang.clone(),
            allowed_languages: cfg.allowed_languages.clone(),
            loading: false,
            error: None,
        }
    }
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/api/sessions", get(list_sessions))
        .route("/api/sessions/{id}", get(get_session))
        .route("/api/settings", get(get_settings).post(post_settings))
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new("static"))
        .with_state(state)
}

async fn get_settings(State(state): State<Arc<AppState>>) -> Response {
    let s = state.settings.read().unwrap().clone();
    Json(json!({
        "model": s.model,
        "target_lang": s.target_lang,
        "allowed_languages": s.allowed_languages,
        "available_models": AVAILABLE_MODELS,
        "loading": s.loading,
        "error": s.error,
    }))
    .into_response()
}

#[derive(Deserialize)]
struct SettingsPatch {
    model: Option<String>,
    target_lang: Option<String>,
}

async fn post_settings(
    State(state): State<Arc<AppState>>,
    Json(patch): Json<SettingsPatch>,
) -> Response {
    if let Some(target) = patch.target_lang {
        let allowed = state.settings.read().unwrap().allowed_languages.clone();
        if !allowed.contains(&target) {
            return (StatusCode::BAD_REQUEST, format!("unsupported language '{target}'"))
                .into_response();
        }
        state.settings.write().unwrap().target_lang = target;
    }

    if let Some(model) = patch.model {
        if !AVAILABLE_MODELS.contains(&model.as_str()) {
            return (StatusCode::BAD_REQUEST, format!("unknown model '{model}'")).into_response();
        }
        let already = {
            let s = state.settings.read().unwrap();
            s.model == model || s.loading
        };
        if !already {
            {
                let mut s = state.settings.write().unwrap();
                s.loading = true;
                s.error = None;
            }
            let st = state.clone();
            let cfg = state.cfg.clone();
            tokio::spawn(async move {
                let result = tokio::task::spawn_blocking(move || {
                    let path = std::path::Path::new(&cfg.models_dir)
                        .join(format!("ggml-{model}.bin"));
                    if !path.exists() {
                        let status = std::process::Command::new("./scripts/fetch-models.sh")
                            .args([model.as_str(), &cfg.models_dir])
                            .status()
                            .map_err(|e| format!("model download failed to start: {e}"))?;
                        if !status.success() {
                            return Err(format!("model download failed for '{model}'"));
                        }
                    }
                    WhisperEngine::new(&path, cfg.whisper_threads)
                        .map(|e| (model, Arc::new(e)))
                        .map_err(|e| format!("model load failed: {e:#}"))
                })
                .await
                .unwrap_or_else(|e| Err(format!("model load task panicked: {e}")));

                let mut s = st.settings.write().unwrap();
                s.loading = false;
                match result {
                    Ok((model, engine)) => {
                        *st.whisper.write().unwrap() = engine;
                        s.model = model;
                    }
                    Err(e) => {
                        tracing::error!("model switch failed: {e}");
                        s.error = Some(e);
                    }
                }
            });
        }
    }

    get_settings(State(state)).await
}

async fn list_sessions(State(state): State<Arc<AppState>>) -> Response {
    match state.db.lock().unwrap().list_sessions() {
        Ok(s) => Json(s).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn get_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<i64>,
) -> Response {
    let db = state.db.lock().unwrap();
    match (db.session_exists(id), db.get_utterances(id)) {
        (Ok(true), Ok(utts)) => {
            let summary = db
                .list_sessions()
                .unwrap_or_default()
                .into_iter()
                .find(|s| s.id == id);
            Json(json!({
                "id": id,
                "started_at": summary.as_ref().map(|s| s.started_at.clone()),
                "title": summary.as_ref().map(|s| s.title.clone()),
                "utterances": utts,
            }))
            .into_response()
        }
        (Ok(false), _) => (StatusCode::NOT_FOUND, "no such session").into_response(),
        (Err(e), _) | (_, Err(e)) => {
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

#[derive(Deserialize)]
struct WsQuery {
    session: Option<i64>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(q): Query<WsQuery>,
    State(state): State<Arc<AppState>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state, q.session))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>, resume: Option<i64>) {
    let session_id = {
        let db = state.db.lock().unwrap();
        match resume {
            Some(id) if db.session_exists(id).unwrap_or(false) => id,
            _ => match db.create_session() {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!("create_session failed: {e:#}");
                    return;
                }
            },
        }
    };

    let (mut sink, mut stream) = socket.split();
    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<i16>>();
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<PipelineEvent>();

    // Snapshot the runtime settings for this session: the engine swap and
    // target-language changes apply to the next connection, not mid-session.
    let deps = {
        let settings = state.settings.read().unwrap();
        let mut cfg = state.cfg.clone();
        cfg.whisper_model = settings.model.clone();
        cfg.target_lang = settings.target_lang.clone();
        PipelineDeps {
            whisper: state.whisper.read().unwrap().clone(),
            db: state.db.clone(),
            cfg,
        }
    };
    std::thread::spawn(move || run_pipeline(deps, session_id, audio_rx, event_tx));

    let send_task = tokio::spawn(async move {
        while let Some(ev) = event_rx.recv().await {
            let msg = match serde_json::to_string(&ev) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if sink.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(msg)) = stream.next().await {
        match msg {
            Message::Binary(b) => {
                let samples: Vec<i16> = b
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect();
                if audio_tx.send(samples).is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    drop(audio_tx); // pipeline thread drains and exits
    let _ = send_task.await;
    let _ = state.db.lock().unwrap().end_session(session_id);
    tracing::info!("session {session_id} socket closed");
}
