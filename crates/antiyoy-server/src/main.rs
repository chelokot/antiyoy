#![forbid(unsafe_code)]

use std::net::{IpAddr, SocketAddr};

use antiyoy_protocol::{
    ClientMessage, CreateMatchRequest, CreateMatchResponse, MatchSnapshot, NETWORK_SCHEMA_VERSION,
    ServerMessage,
};
use antiyoy_server::{MatchService, ServiceError};
use anyhow::Result;
use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use futures_util::StreamExt;
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::broadcast;

#[derive(Debug, Parser)]
#[command(name = "antiyoy-server", version, about)]
struct Cli {
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[derive(Debug, Serialize)]
struct Health {
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: &'static str,
    message: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let address = SocketAddr::new(cli.host, cli.port);
    let listener = TcpListener::bind(address).await?;
    axum::serve(listener, router(MatchService::new()))
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

fn router(service: MatchService) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/matches", post(create_match))
        .route("/v1/matches/{match_id}", get(match_state))
        .route("/v1/matches/{match_id}/replay", get(match_replay))
        .route("/v1/matches/{match_id}/watch", get(watch_match))
        .with_state(service)
}

async fn health() -> Json<Health> {
    Json(Health { status: "ok" })
}

async fn create_match(
    State(service): State<MatchService>,
    Json(request): Json<CreateMatchRequest>,
) -> Result<Json<CreateMatchResponse>, ApiError> {
    Ok(Json(
        blocking_match(move || service.create_match(&request)).await?,
    ))
}

async fn match_state(
    State(service): State<MatchService>,
    Path(match_id): Path<String>,
) -> Result<Json<MatchSnapshot>, ApiError> {
    Ok(Json(
        blocking_match(move || service.snapshot(&match_id)).await?,
    ))
}

async fn match_replay(
    State(service): State<MatchService>,
    Path(match_id): Path<String>,
) -> Result<Response, ApiError> {
    let replay = blocking_match(move || service.replay(&match_id)).await?;
    Ok((
        [(header::CONTENT_TYPE, "application/vnd.antiyoy.replay")],
        Body::from(replay),
    )
        .into_response())
}

async fn watch_match(
    websocket: WebSocketUpgrade,
    State(service): State<MatchService>,
    Path(match_id): Path<String>,
) -> Result<Response, ApiError> {
    let subscription_service = service.clone();
    let subscription_match = match_id.clone();
    let (snapshot, updates) =
        blocking_match(move || subscription_service.subscribe(&subscription_match)).await?;
    Ok(websocket
        .on_upgrade(move |socket| socket_loop(socket, service, match_id, snapshot, updates)))
}

async fn socket_loop(
    mut socket: WebSocket,
    service: MatchService,
    match_id: String,
    snapshot: MatchSnapshot,
    mut updates: broadcast::Receiver<MatchSnapshot>,
) {
    let mut seat = None;
    if send_message(&mut socket, ServerMessage::Snapshot(snapshot))
        .await
        .is_err()
    {
        return;
    }
    loop {
        tokio::select! {
            incoming = socket.next() => {
                let Some(Ok(message)) = incoming else { return };
                match message {
                    Message::Text(text) => {
                        let response = match serde_json::from_str::<ClientMessage>(&text) {
                            Ok(ClientMessage::Authenticate { schema_version, seat: requested, token }) => {
                                if schema_version == NETWORK_SCHEMA_VERSION {
                                    match service.authorize(&match_id, requested, &token) {
                                        Ok(()) => {
                                            seat = Some((requested, token));
                                            ServerMessage::Authenticated { seat: requested }
                                        }
                                        Err(error) => server_error(&error),
                                    }
                                } else {
                                    ServerMessage::Error {
                                        code: "invalid_request".into(),
                                        message: format!("network schema {schema_version} is unsupported"),
                                    }
                                }
                            }
                            Ok(ClientMessage::Submit(submission)) if seat.is_some() => {
                                let (seat, token) = seat.as_ref().expect("authenticated seat exists");
                                let action_service = service.clone();
                                let action_match = match_id.clone();
                                let action_token = token.clone();
                                let action_seat = *seat;
                                match blocking_match(move || action_service.submit(
                                    &action_match,
                                    action_seat,
                                    &action_token,
                                    submission,
                                )).await {
                                    Ok(_) => continue,
                                    Err(error) => server_error(&error),
                                }
                            }
                            Ok(ClientMessage::Submit(_)) => ServerMessage::Error {
                                code: "spectator".into(),
                                message: "authenticate before submitting actions".into(),
                            },
                            Err(error) => ServerMessage::Error {
                                code: "invalid_message".into(),
                                message: error.to_string(),
                            },
                        };
                        if send_message(&mut socket, response).await.is_err() { return }
                    }
                    Message::Close(_) => return,
                    Message::Ping(payload) => {
                        if socket.send(Message::Pong(payload)).await.is_err() { return }
                    }
                    Message::Pong(_) | Message::Binary(_) => {}
                }
            }
            update = updates.recv() => {
                let snapshot = match update {
                    Ok(snapshot) => snapshot,
                    Err(broadcast::error::RecvError::Lagged(_)) => match blocking_match({
                        let service = service.clone();
                        let match_id = match_id.clone();
                        move || service.snapshot(&match_id)
                    }).await {
                        Ok(snapshot) => snapshot,
                        Err(_) => return,
                    },
                    Err(broadcast::error::RecvError::Closed) => return,
                };
                if send_message(&mut socket, ServerMessage::Snapshot(snapshot)).await.is_err() { return }
            }
        }
    }
}

async fn send_message(socket: &mut WebSocket, message: ServerMessage) -> Result<(), axum::Error> {
    let encoded = serde_json::to_string(&message).expect("network messages are JSON serializable");
    socket.send(Message::Text(encoded.into())).await
}

fn server_error(error: &ServiceError) -> ServerMessage {
    let code = error_code(error).to_owned();
    ServerMessage::Error {
        code,
        message: error.to_string(),
    }
}

struct ApiError(ServiceError);

impl From<ServiceError> for ApiError {
    fn from(error: ServiceError) -> Self {
        Self(error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            ServiceError::InvalidRequest(_) | ServiceError::IllegalAction(_) => {
                StatusCode::UNPROCESSABLE_ENTITY
            }
            ServiceError::NotFound(_) => StatusCode::NOT_FOUND,
            ServiceError::Unauthorized => StatusCode::UNAUTHORIZED,
            ServiceError::StaleRevision { .. }
            | ServiceError::WrongTurn { .. }
            | ServiceError::Finished => StatusCode::CONFLICT,
            ServiceError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let code = error_code(&self.0);
        (
            status,
            Json(ErrorBody {
                code,
                message: self.0.to_string(),
            }),
        )
            .into_response()
    }
}

fn error_code(error: &ServiceError) -> &'static str {
    match error {
        ServiceError::InvalidRequest(_) => "invalid_request",
        ServiceError::NotFound(_) => "not_found",
        ServiceError::Unauthorized => "unauthorized",
        ServiceError::StaleRevision { .. } => "stale_revision",
        ServiceError::WrongTurn { .. } => "wrong_turn",
        ServiceError::Finished => "finished",
        ServiceError::IllegalAction(_) => "illegal_action",
        ServiceError::Internal(_) => "internal",
    }
}

async fn blocking_match<T>(
    operation: impl FnOnce() -> Result<T, ServiceError> + Send + 'static,
) -> Result<T, ServiceError>
where
    T: Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| ServiceError::Internal(error.to_string()))?
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
