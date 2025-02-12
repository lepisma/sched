use std::path::PathBuf;
use ollama_rs::{generation::completion::request::GenerationRequest, Ollama};
use std::{fs, io::Write};
use anyhow::Result;

use clap::Parser;

const MODEL: &str = "deepseek-r1:1.5b";

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

async fn setup_model() -> Result<()> {
    let ollama = Ollama::default();
    ollama.pull_model(MODEL.to_string(), true).await?;
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

async fn generate_output(task: &Task) -> TaskOutput {
    let ollama = Ollama::default();
    let res = ollama.generate(GenerationRequest::new(MODEL.to_string(), &task.prompt)).await.unwrap();

    TaskOutput {
        response: res.response,
        model: MODEL.to_string(),
    }
}

fn generate_file_name(task: &Task, task_output: &TaskOutput) -> String {
    format!("{}.{}.output", task.id, task_output.model)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    setup_model().await?;

    let tasks: Vec<Task> = read_tasks(&args.dir_path)
        .into_iter()
        .filter(|t| read_outputs(&args.dir_path, t).is_empty())
        .collect();

    log::info!("Found {} task(s) to do.", tasks.len());

    if !tasks.is_empty() {
        let bar = indicatif::ProgressBar::new(tasks.len() as u64);

        for task in tasks {
            let task_output = generate_output(&task).await;
            let file_name = generate_file_name(&task, &task_output);

            let mut file = fs::File::create(args.dir_path.join(file_name))?;
            file.write_all(task_output.response.as_bytes())?;

            bar.inc(1);
        }

        bar.finish();
    }

    Ok(())
}
