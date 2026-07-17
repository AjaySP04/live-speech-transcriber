use futures_util::{SinkExt, StreamExt};
use tarjuman::{config::Config, db::Db, server, whisper::WhisperEngine};
use std::sync::{Arc, Mutex};
use tokio_tungstenite::tungstenite::Message;

async fn spawn_server() -> (String, Arc<server::AppState>) {
    assert!(
        std::path::Path::new("models/ggml-tiny.bin").exists(),
        "run scripts/fetch-models.sh tiny"
    );
    let mut cfg = Config::default();
    cfg.whisper_model = "tiny".into();
    let whisper = Arc::new(WhisperEngine::new(&cfg.whisper_model_path(), 4).unwrap());
    let db = Arc::new(Mutex::new(Db::open_in_memory().unwrap()));
    let state = Arc::new(server::AppState {
        whisper: std::sync::RwLock::new(whisper),
        db,
        settings: std::sync::RwLock::new(server::AppState::runtime_settings(&cfg)),
        cfg,
    });
    let app = server::router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("127.0.0.1:{}", addr.port()), state)
}

#[tokio::test]
async fn healthz_and_empty_sessions() {
    let (addr, _state) = spawn_server().await;
    let body = http_get(&format!("http://{addr}/healthz")).await;
    assert_eq!(body, "ok");
    let body = http_get(&format!("http://{addr}/api/sessions")).await;
    assert_eq!(body.trim(), "[]");
}

#[tokio::test]
async fn ws_creates_session_and_reports_it() {
    let (addr, state) = spawn_server().await;
    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws"))
        .await
        .unwrap();
    // first message must be session_started
    let msg = tokio::time::timeout(std::time::Duration::from_secs(10), ws.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(msg.to_text().unwrap()).unwrap();
    assert_eq!(v["type"], "session_started");
    let sid = v["session_id"].as_i64().unwrap();

    // stream a little silence, then close; session must exist and be ended
    ws.send(Message::Binary(vec![0u8; 8192].into())).await.unwrap();
    ws.close(None).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert!(state.db.lock().unwrap().session_exists(sid).unwrap());
}

// Minimal HTTP/1.0 GET over std TcpStream — avoids adding an HTTP client dependency.
async fn http_get(url: &str) -> String {
    let url = url.strip_prefix("http://").unwrap().to_string();
    let (host, path) = url.split_once('/').map(|(h, p)| (h.to_string(), format!("/{p}"))).unwrap();
    tokio::task::spawn_blocking(move || {
        use std::io::{Read, Write};
        let mut s = std::net::TcpStream::connect(&host).unwrap();
        write!(s, "GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n").unwrap();
        let mut buf = String::new();
        s.read_to_string(&mut buf).unwrap();
        buf.split("\r\n\r\n").nth(1).unwrap_or("").to_string()
    })
    .await
    .unwrap()
}
