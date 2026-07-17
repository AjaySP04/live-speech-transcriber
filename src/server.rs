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
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex};
use tower_http::services::ServeDir;

pub struct AppState {
    pub whisper: Arc<WhisperEngine>,
    pub db: Arc<Mutex<Db>>,
    pub cfg: Config,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/api/sessions", get(list_sessions))
        .route("/api/sessions/{id}", get(get_session))
        .route("/ws", get(ws_handler))
        .fallback_service(ServeDir::new("static"))
        .with_state(state)
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

    let deps = PipelineDeps {
        whisper: state.whisper.clone(),
        db: state.db.clone(),
        cfg: state.cfg.clone(),
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
