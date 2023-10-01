use std::{
    io::Write,
    ops::ControlFlow,
    sync::{Arc, RwLock},
    thread,
};

use anyhow::Result;
use tracing::metadata::LevelFilter;

use config::TrainEmbeddingConfig;
use messages::{TrainerHandleSender, TrainerMessage, YieldState};
use model::EmbedModel;

mod bounded;
mod config;
mod messages;
mod model;
mod training;

#[cfg(not(feature = "thread"))]
fn main() -> Result<()> {
    use std::cell::RefCell;

    configure_logging();
    let (config, resumed_state) = parse_cli_args()?;

    let on_yield: Rc<RefCell<Option<Rc<dyn Fn() -> ()>>>> = Rc::new(RefCell::new(None));

    let config_clone = config.clone();
    let on_yield_clone = on_yield.clone();
    let (tx, handle) =
        messages::TrainerHandle::new(move |embedding: &EmbedModel, msg: TrainerMessage| {
            match &msg {
                TrainerMessage::Yield => {
                    if let Some(on_yield) = on_yield_clone.borrow().as_ref() {
                        (**on_yield)()
                    }
                }
                _ => {}
            }
            let handle_action = msg.apply(embedding, &config_clone);
            handle_action
        });

    if let Some((snapshot, state)) = resumed_state {
        tx.send(TrainerMessage::ReplaceEmbeddingState(snapshot, state))?;
    }

    if let Some(repl) = &config.repl {
        for c in repl.chars() {
            parse_repl_char(c, &tx, &config).expect("config repl character caused startup to halt");
        }
    }

    // let config_clone = config.clone();
    // let tx1 = tx.clone();
    on_yield.replace(Some(Rc::new(move || {
        // let tx = tx1.clone();
        // let config_clone = config_clone.clone();
        // block_on_key_press(move |c| parse_repl_char(c, &tx, &config_clone), &tx1)
    })));

    training::setup_and_train_model_v2(config, handle);
    Ok(())
}

#[cfg(feature = "thread")]
fn main() -> Result<()> {
    use crate::messages::TrainerHandleActions;

    configure_logging();
    let (config, resumed_state) = parse_cli_args()?;
    let latest_state = Arc::new(RwLock::new(None));
    parse_wait_for_input_repl(&config);

    let latest_state_clone = latest_state.clone();
    let config_clone = config.clone();
    let (tx, handle) =
        messages::TrainerHandle::new(move |model: &EmbedModel, msg: TrainerMessage| {
            let handle_action = match msg {
                TrainerMessage::Yield(state) => {
                    latest_state_clone.write().unwrap().replace(state);
                    return TrainerHandleActions::Nothing;
                }
                msg => msg.apply(model, &config_clone),
            };
            handle_action
        });

    let config_clone = config.clone();
    let thread = thread::spawn(move || {
        let _embedding = training::setup_and_train_model_v2(config_clone, handle);
    });

    if let Some((snapshot, state)) = resumed_state {
        tx.send(TrainerMessage::ReplaceEmbeddingState(snapshot, state))?;
    }

    let latest_state_clone = latest_state.clone();
    let config_clone = config.clone();
    let ui_thread = thread::spawn(move || {
        if let Some(repl) = config.repl {
            for c in repl.chars() {
                let control_flow =
                    parse_repl_char(c, &tx, &config_clone, YieldStateAccessor::default())
                        .expect("config repl character caused startup to halt");

                if let ControlFlow::Break(()) = control_flow {
                    return;
                }
            }
        }

        let tx1 = tx.clone();
        let latest_state_ref = &latest_state_clone;
        block_on_key_press(
            move |c| parse_repl_char(c, &tx, &config_clone, latest_state_ref.into()),
            &tx1,
        );
    });

    thread.join().unwrap();
    ui_thread.join().unwrap();

    Ok(())
}

fn parse_repl_char(
    c: char,
    tx: &TrainerHandleSender<TrainerMessage>,
    config: &TrainEmbeddingConfig,
    latest_state: YieldStateAccessor,
) -> Result<ControlFlow<()>> {
    match c {
        'r' => tx.send(TrainerMessage::PrintStatus)?,
        'o' => tx.send(TrainerMessage::PrintTrainingStatus)?,
        'R' => tx.send(TrainerMessage::SuppressAutoPrintStatus)?,
        'P' => tx.send(TrainerMessage::PrintEachRoundNumber)?,
        'a' => tx.send(TrainerMessage::TogglePrintAllStatus)?,
        'x' => tx.send(TrainerMessage::ReloadFromSnapshot)?,
        'z' => tx.send(TrainerMessage::ForceSnapshot)?,
        'g' => tx.send(TrainerMessage::PlotTrainingLossGraph)?,
        'G' => tx.send(TrainerMessage::PlotHeatMapGraphs)?, //tmp
        'n' => tx.send(TrainerMessage::PredictRandomPhrase)?,
        'w' => tx.send(TrainerMessage::WriteStatsToDisk)?,
        'W' => tx.send(TrainerMessage::WriteEmbeddingTsvToDisk)?,
        's' => tx.send(TrainerMessage::WriteModelToDisk)?,
        ',' => tx.send(TrainerMessage::MultiplyLearnRateBy(0.5))?,
        '.' => tx.send(TrainerMessage::MultiplyLearnRateBy(2.0))?,
        'e' => tx.send(TrainerMessage::IncreaseMaxRounds(config.training_rounds))?,
        '/' => tx.send(TrainerMessage::UnpauseForIterations(1))?,
        '_' => tx.send(prompt("iterations", |x| match x.parse::<usize>() {
            Ok(n) => TrainerMessage::UnpauseForIterations(n),
            Err(_) => TrainerMessage::NoOp,
        }))?,
        'p' => tx.send(TrainerMessage::TogglePause)?,
        'c' => tx.send(TrainerMessage::PrintConfig)?,
        'l' => tx.send(prompt("output_label", |x| {
            TrainerMessage::RenameOutputLabel(x)
        }))?,
        'q' => {
            tx.send(TrainerMessage::Halt)?;
            return Ok(ControlFlow::Break(()));
        }
        '?' => {
            if let Some(latest_state) = latest_state.read() {
                println!(r"Trainer state: Round {}", latest_state.round);
            }
        }
        'h' => {
            println!(
                r"Trainer help: 
                        'r' => report status (testing set)
                        'o' => report status (training set)
                        'R' => supress auto-report status
                        'P' => toggle always report round number
                        'o' => toggle always run testing report 
                        'n' => print new random phrase
                        'x' => reload from auto-save snapshot
                        'z' => force snapshot
                        'G' => open next heatmap plots in browser (tmp)
                        'g' => open training plot in browser
                        'w' => write stats to disk
                        'W' => write embedding tsv to disk
                        's' => save model to disk
                        ',' => divide learn rate by 2
                        '.' => multiply learn rate by 2
                        'e' => extend training rounds
                        'l' => rename output label (applies to subsequent save model operations)
                        'c' => display config
                        'h' => display help
                        '/' => pause after a single iteration
                        'p' => toggle pause
                        'q' => quit
                        "
            );
        }
        _ => (),
    }
    Ok(ControlFlow::Continue(()))
}

fn parse_wait_for_input_repl(config: &TrainEmbeddingConfig) {
    if let Some(c) = config.repl.as_ref().and_then(|x| x.chars().next()) {
        if c == '#' {
            let pid = std::process::id();
            println!("Waiting for input... (pid = {pid})");
            let stdin = std::io::stdin();
            let mut tmp = String::new();
            stdin.read_line(&mut tmp).unwrap();
        }
    }
}

#[cfg(feature = "cli")]
fn block_on_key_press<F: Fn(char) -> Result<ControlFlow<()>>>(
    key_press_callback: F,
    tx: &TrainerHandleSender<TrainerMessage>,
) {
    use crossterm::event::{self, Event, KeyCode::Char, KeyEvent};

    loop {
        match event::poll(std::time::Duration::from_secs(10)) {
            Ok(true) => {}
            Ok(false) => {
                if let Err(_) = tx.send(TrainerMessage::NoOp) {
                    return;
                }
                continue;
            }
            Err(_) => return,
        };
        if let Event::Key(KeyEvent { code: Char(c), .. }) = event::read().unwrap() {
            if let Err(_) | Ok(ControlFlow::Break(())) = key_press_callback(c) {
                return;
            }
        }
    }
}

#[cfg(not(feature = "cli"))]
fn block_on_key_press<F: Fn(char) -> Result<ControlFlow<()>>>(
    key_press_callback: F,
    tx: &TrainerHandleSender<TrainerMessage>,
) {
    use std::io::BufRead;

    let stdin = std::io::stdin();
    loop {
        let mut stdin = stdin.lock();
        let buffer = match stdin.fill_buf() {
            Ok(buffer) if !buffer.is_empty() => buffer,
            Ok(_) => {
                if let Err(_) = tx.send(TrainerMessage::NoOp) {
                    return;
                }
                continue;
            }
            Err(_) => return,
        };

        let n = buffer.len();

        let buffer: String = std::str::from_utf8(buffer).unwrap().to_string();
        stdin.consume(n);
        drop(stdin);

        for c in buffer.trim().chars() {
            if let Err(_) | Ok(ControlFlow::Break(())) = key_press_callback(c) {
                return;
            }
        }
    }
}

fn prompt<F: Fn(String) -> TrainerMessage>(
    var_name: &str,
    message_factory_fn: F,
) -> TrainerMessage {
    let mut input = String::new();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    stdout
        .write_fmt(format_args!(
            "Set new value for '{var_name}' [or type '.cancel']: "
        ))
        .unwrap();
    stdout.flush().unwrap();

    match stdin.read_line(&mut input) {
        Ok(_) if input.trim() != ".cancel" => {
            println!("Setting new value of '{var_name}'. Exiting edit mode...");
            message_factory_fn(input.trim().to_string())
        }
        _ => {
            println!("Retaining previous value of '{var_name}'. Exiting edit mode...");
            TrainerMessage::NoOp
        }
    }
}

#[derive(Default)]
struct YieldStateAccessor(Option<Arc<RwLock<Option<YieldState>>>>);

impl From<&Arc<RwLock<Option<YieldState>>>> for YieldStateAccessor {
    fn from(value: &Arc<RwLock<Option<YieldState>>>) -> Self {
        Self(Some(value.clone()))
    }
}

impl YieldStateAccessor {
    fn read(&self) -> Option<YieldState> {
        self.0
            .as_ref()
            .and_then(|x| x.read().ok().and_then(|x| x.clone()))
    }
}

fn configure_logging() {
    let subscriber = tracing_subscriber::fmt()
        .compact()
        .with_target(false)
        .with_max_level(LevelFilter::INFO)
        // .with_span_events(FmtSpan::CLOSE)
        .finish();

    tracing::subscriber::set_global_default(subscriber).unwrap();
}

fn parse_cli_args() -> Result<(
    TrainEmbeddingConfig,
    Option<(String, messages::TrainerStateMetadata)>,
)> {
    use clap::{Args, FromArgMatches};

    let cli = clap::Command::new("embed").disable_help_flag(true).arg(
        clap::Arg::new("help")
            .long("help")
            .action(clap::ArgAction::Help),
    );
    let cli = config::Cli::augment_args(cli);
    let matches = cli.get_matches();
    let cli = config::Cli::from_arg_matches(&matches)
        .map_err(|err| err.exit())
        .unwrap();

    let (mut config, resumed_state) = match cli.command() {
        config::Command::TrainEmbedding(config) => (config, None),
        config::Command::JsonTrainEmbedding(config) => (
            load_resume_config(config.trainer_config, config.resume_command),
            None,
        ),
        config::Command::LoadEmbedding(config) => load_embedding(config)?,
    };

    init_train_config(&mut config);

    Ok((config, resumed_state))
}

fn init_train_config(config: &mut config::TrainEmbeddingConfig) {
    if let (Some(output_label), true) = (&config.output_label, config.output_label_append_details) {
        let deep_layer_counts = config
            .hidden_deep_layer_nodes
            .as_ref()
            .map(|nodes| format!("x{}", nodes.replace(",", "x")))
            .unwrap_or_default();

        config.output_label = Some(format!(
            "{}-e{}-L{}{}",
            &output_label, &config.embedding_size, &config.hidden_layer_nodes, &deep_layer_counts
        ));

        config.output_label_append_details = false;
    }
}

fn load_embedding(
    load_config: config::LoadEmbeddingConfig,
) -> Result<(
    TrainEmbeddingConfig,
    Option<(String, messages::TrainerStateMetadata)>,
)> {
    let file_path = &load_config.file_path;
    let (snapshot, config, state) = training::writer::read_model_from_disk(file_path)?;
    let load_config = load_config.config;

    Ok((
        load_resume_config(config, load_config),
        Some((snapshot, state)),
    ))
}

fn load_resume_config(
    mut config: TrainEmbeddingConfig,
    resume_config: config::ResumeEmbeddingConfig,
) -> TrainEmbeddingConfig {
    config.pause_on_start = true;
    config.repl = resume_config.repl;
    // always overwrite repl

    if let Some(hidden_layer_nodes) = resume_config.hidden_layer_nodes {
        config.hidden_layer_nodes = hidden_layer_nodes;
    }
    if let Some(hidden_deep_layer_nodes) = resume_config.hidden_deep_layer_nodes {
        config.hidden_deep_layer_nodes = Some(hidden_deep_layer_nodes);
    }
    if let Some(input_stride_width) = resume_config.input_stride_width {
        config.input_stride_width = input_stride_width;
    }
    if let Some(batch_size) = resume_config.batch_size {
        config.batch_size = batch_size;
    }
    if let Some(input_txt_path) = resume_config.input_txt_path {
        config.input_txt_path = Some(input_txt_path);
    }
    if let Some(phrase_test_set_max_tokens) = resume_config.phrase_test_set_max_tokens {
        config.phrase_test_set_max_tokens = Some(phrase_test_set_max_tokens);
    }
    if resume_config.disable_sample_from_pattern {
        config.sample_from_pattern = None;
    }
    if resume_config.force_continue {
        config.quit_on_complete = false;
    }
    config
}
