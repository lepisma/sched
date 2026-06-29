use std::path::PathBuf;
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use std::{fs, io::Write};
use anyhow::Result;
use std::fs::OpenOptions;

use clap::Parser;

const DEFAULT_MODEL: &str = "deepseek-r1:1.5b";

#[derive(clap::Parser, Debug)]
struct Args {
    dir_path: PathBuf,
}

// A single prompt task for an LLM to run
#[derive(Debug)]
struct Task {
    id: String,
    prompt: String
}

#[derive(Debug)]
struct TaskOutput {
    response: String,
    model: String,
}

// Return the model name to use, falling back to the hardcoded default
fn get_model_name() -> String {
    std::env::var("SCHED_MODEL").unwrap_or(DEFAULT_MODEL.to_string())
}

async fn setup_model() -> Result<()> {
    let ollama = Ollama::default();
    ollama.pull_model(get_model_name(), true).await?;
    Ok(())
}

// Read a list of task files from given directory
fn read_tasks(dir_path: &PathBuf) -> Vec<Task> {
    let mut tasks = Vec::new();

    for entry in std::fs::read_dir(dir_path).unwrap() {
        let path = entry.unwrap().path();

        if path.extension().and_then(|ext| ext.to_str()) == Some("task") {
            if let Some(id) = path.file_stem().and_then(|s| s.to_str()) {
                tasks.push(Task {
                    id: id.to_string(),
                    prompt: fs::read_to_string(&path).unwrap(),
                });
            }
        }
    }

    tasks
}

fn read_outputs(dir_path: &PathBuf, task: &Task) -> Vec<TaskOutput> {
    let mut outputs = Vec::new();

    for entry in std::fs::read_dir(dir_path).unwrap() {
        let path = entry.unwrap().path();

        if path.extension().and_then(|ext| ext.to_str()) == Some("output") {
            if let Some(file_name) = path.file_stem().and_then(|s| s.to_str()) {
                if let Some((id, model_name)) = parse_output_filename(file_name) {
                    if id == task.id {
                        outputs.push(TaskOutput {
                            response: fs::read_to_string(&path).unwrap().to_string(),
                            model: model_name.to_string(),
                        });
                    }
                }
            }
        }
    }

    outputs
}

fn parse_output_filename(file_name: &str) -> Option<(&str, &str)> {
    let parts: Vec<&str> = file_name.splitn(2, ".").collect();
    if parts.len() == 2 {
        Some((parts[0], parts[1]))
    } else {
        None
    }
}

fn lock_path(dir_path: &PathBuf, task: &Task) -> PathBuf {
    dir_path.join(format!("{}.lock", task.id))
}

// Returns true if this process successfully claimed the task
fn try_claim_task(dir_path: &PathBuf, task: &Task) -> bool {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(lock_path(dir_path, task))
        .is_ok()
}

fn release_task(dir_path: &PathBuf, task: &Task) {
    let _ = fs::remove_file(lock_path(dir_path, task));
}

async fn generate_output(task: &Task) -> TaskOutput {
    let ollama = Ollama::default();
    let model = get_model_name();
    let res = ollama.generate(GenerationRequest::new(model.clone(), &task.prompt)).await.unwrap();

    TaskOutput {
        response: res.response,
        model,
    }
}

fn generate_file_name(task: &Task, task_output: &TaskOutput) -> String {
    format!("{}.{}.output", task.id, task_output.model)
}

async fn is_machine_busy(duration: usize) -> bool {
    let mut sys = sysinfo::System::new_all();
    let mut total_usage = 0.0;

    for _ in 0..duration {
        sys.refresh_cpu_all();
        let usage: f32 = sys.cpus().iter().map(|cpu| cpu.cpu_usage()).sum::<f32>() / sys.cpus().len() as f32;
        total_usage += usage;
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }

    let average_usage = total_usage / duration as f32;
    log::debug!("Average usage in last {} seconds: {:.2}%", duration, average_usage);

    average_usage > 30.0
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    setup_model().await?;

    loop {
        let tasks: Vec<Task> = read_tasks(&args.dir_path)
            .into_iter()
            .filter(|t| read_outputs(&args.dir_path, t).is_empty())
            .filter(|t| !lock_path(&args.dir_path, t).exists())
            .collect();

        log::info!("Found {} task(s) to do.", tasks.len());

        if !tasks.is_empty() {
            let bar = indicatif::ProgressBar::new(tasks.len() as u64);

            for task in tasks {
                loop {
                    log::info!("Waiting before working on task");
                    // Wait for some time between tasks and check if the machine
                    // is free. We are in no hurry.
                    if is_machine_busy(10).await {
                        log::info!("Machine busy...");
                    } else {
                        log::info!("Running task");
                        break;
                    }
                }

                if !try_claim_task(&args.dir_path, &task) {
                    log::info!("Task {} already claimed by another worker, skipping", task.id);
                    bar.inc(1);
                    continue;
                }

                let task_output = generate_output(&task).await;
                let file_name = generate_file_name(&task, &task_output);

                let mut file = fs::File::create(args.dir_path.join(file_name))?;
                file.write_all(task_output.response.as_bytes())?;
                release_task(&args.dir_path, &task);

                bar.inc(1);
            }

            bar.finish();
        }

        // Wait for some time and then re-check tasks
        let wait_time_m = 10;
        log::info!("Waiting for {} minutes", wait_time_m);
        tokio::time::sleep(tokio::time::Duration::from_secs(wait_time_m * 60)).await;
    }
}
