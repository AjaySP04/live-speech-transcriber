use live_transcriber::{config::Config, db::Db, server, whisper::WhisperEngine};
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().init();
    let cfg = Config::load("config.toml")?;
    std::fs::create_dir_all(&cfg.data_dir)?;

    let whisper_path = cfg.whisper_model_path();
    anyhow::ensure!(
        whisper_path.exists(),
        "whisper model missing at {whisper_path:?} — run: scripts/fetch-models.sh {}",
        cfg.whisper_model
    );
    anyhow::ensure!(
        cfg.speaker_model_path().exists(),
        "speaker model missing — run: scripts/fetch-models.sh"
    );

    tracing::info!("loading whisper model {:?}…", whisper_path);
    let whisper = Arc::new(WhisperEngine::new(&whisper_path, cfg.whisper_threads)?);
    let db = Arc::new(Mutex::new(Db::open(&format!("{}/transcriber.db", cfg.data_dir))?));

    let state = Arc::new(server::AppState { whisper, db, cfg: cfg.clone() });
    let app = server::router(state);
    let addr: std::net::SocketAddr = format!("0.0.0.0:{}", cfg.port).parse()?;

    if cfg.tls {
        let cert_path = format!("{}/cert.pem", cfg.data_dir);
        let key_path = format!("{}/key.pem", cfg.data_dir);
        if !std::path::Path::new(&cert_path).exists() {
            let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()])?;
            std::fs::write(&cert_path, cert.cert.pem())?;
            std::fs::write(&key_path, cert.key_pair.serialize_pem())?;
            tracing::info!("generated self-signed cert in {}", cfg.data_dir);
        }
        let tls = axum_server::tls_rustls::RustlsConfig::from_pem_file(&cert_path, &key_path).await?;
        tracing::info!("listening on https://{addr}");
        axum_server::bind_rustls(addr, tls)
            .serve(app.into_make_service())
            .await?;
    } else {
        tracing::info!("listening on http://{addr}");
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
    }
    Ok(())
}
