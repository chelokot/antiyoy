use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use antiyoy_eval::League;
use antiyoy_protocol::{CreateMatchRequest, Replay};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub(crate) const ROOM_STORAGE_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) struct StoredRoom {
    pub schema_version: u16,
    pub id: String,
    pub request: CreateMatchRequest,
    pub human_token_hashes: [Option<[u8; 32]>; 2],
    pub replay: Replay,
    pub rated: bool,
    pub duplicate: bool,
}

pub(crate) struct Storage {
    root: PathBuf,
}

#[derive(Debug, Error)]
pub(crate) enum StorageError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),
    #[error("league JSON error: {0}")]
    LeagueJson(#[from] serde_json::Error),
    #[error("stored league is invalid: {0}")]
    InvalidLeague(String),
    #[error("stored room filename {0} is invalid")]
    InvalidFilename(String),
    #[error("stored room {stored} was found in file for {filename}")]
    IdentityMismatch { filename: String, stored: String },
}

impl Storage {
    pub fn new(root: &Path) -> Result<Self, StorageError> {
        fs::create_dir_all(root)?;
        Ok(Self {
            root: root.to_owned(),
        })
    }

    pub fn load(&self) -> Result<Vec<StoredRoom>, StorageError> {
        let mut files = fs::read_dir(&self.root)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<Result<Vec<_>, _>>()?;
        files.sort();
        let mut rooms = Vec::new();
        let mut removed_temporary = false;
        for path in files {
            let temporary = path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == "league.json.tmp" || name.ends_with(".room.tmp"));
            if path.is_file() && temporary {
                fs::remove_file(path)?;
                removed_temporary = true;
                continue;
            }
            if path.extension().and_then(|extension| extension.to_str()) != Some("room") {
                continue;
            }
            let filename = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| StorageError::InvalidFilename(path.display().to_string()))?;
            let bytes = fs::read(&path)?;
            let room: StoredRoom = postcard::from_bytes(&bytes)?;
            if room.id != filename {
                return Err(StorageError::IdentityMismatch {
                    filename: filename.into(),
                    stored: room.id,
                });
            }
            rooms.push(room);
        }
        if removed_temporary {
            self.sync_root()?;
        }
        Ok(rooms)
    }

    pub fn save(&self, room: &StoredRoom) -> Result<(), StorageError> {
        let bytes = postcard::to_allocvec(room)?;
        let temporary = self.root.join(format!("{}.room.tmp", room.id));
        let destination = self.root.join(format!("{}.room", room.id));
        self.write_atomic(&temporary, &destination, &bytes)
    }

    pub fn load_league(&self) -> Result<League, StorageError> {
        let path = self.root.join("league.json");
        if !path.exists() {
            return Ok(League::default());
        }
        let league: League = serde_json::from_slice(&fs::read(path)?)?;
        league
            .validate()
            .map_err(|error| StorageError::InvalidLeague(error.to_string()))?;
        Ok(league)
    }

    pub fn save_league(&self, league: &League) -> Result<(), StorageError> {
        league
            .validate()
            .map_err(|error| StorageError::InvalidLeague(error.to_string()))?;
        let bytes = serde_json::to_vec_pretty(league)?;
        self.write_atomic(
            &self.root.join("league.json.tmp"),
            &self.root.join("league.json"),
            &bytes,
        )
    }

    fn write_atomic(
        &self,
        temporary: &Path,
        destination: &Path,
        bytes: &[u8],
    ) -> Result<(), StorageError> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(temporary, destination)?;
        self.sync_root()
    }

    pub fn delete(&self, id: &str) -> Result<(), StorageError> {
        fs::remove_file(self.root.join(format!("{id}.room")))?;
        self.sync_root()
    }

    fn sync_root(&self) -> Result<(), StorageError> {
        File::open(&self.root)?.sync_all()?;
        Ok(())
    }
}
