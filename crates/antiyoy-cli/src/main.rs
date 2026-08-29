use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "antiyoy", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Version,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Version => println!("{}", antiyoy_core::ENGINE_VERSION),
    }
}
