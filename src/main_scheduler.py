import os
import json
import subprocess
from itertools import product
import time
import shutil
import glob

def setup_job_directory(exp_name):
    """Setup job directory with clean copy of source files"""
    job_dir = f"jobs/{exp_name}"
    
    # Remove existing directory if it exists
    if os.path.exists(job_dir):
        shutil.rmtree(job_dir)
    
    # Create fresh directory
    os.makedirs(job_dir)

def run_experiment_1(slurm_args):
    """Sweep over reasoning steps with fixed learning rate and lora dim"""
    setup_job_directory("exp1")
    
    reasoning_steps_values = [5, 10, 20, 30, 50]
    learning_rate = 1e-3
    lora_dim = 32
    
    jobs = []
    for job_id, steps in enumerate(reasoning_steps_values):
        job_dir = f"jobs/exp1/job{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        
        # Copy all python files to job directory
        for py_file in glob.glob("*.py"):
            shutil.copy2(py_file, job_dir)
        
        cmd = [
            "cd", job_dir, "&&",
            "srun",
            *[f"{k}={v}" if v else k for k,v in slurm_args.items()],
            "python", "main_job.py",
            "--learning_rate", str(learning_rate),
            "--num_epochs", "10",
            "--reasoning_steps", str(steps),
            "--lora_dim", str(lora_dim)
        ]
        
        process = subprocess.Popen(" ".join(cmd), shell=True)
        jobs.append((process, job_dir, {
            "job_id": job_id,
            "reasoning_steps": steps,
            "learning_rate": learning_rate,
            "lora_dim": lora_dim
        }))
    
    return jobs

def run_experiment_2(slurm_args):
    """Sweep over learning rates with fixed reasoning steps and lora dim"""
    setup_job_directory("exp2")
    
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
    reasoning_steps = 30
    lora_dim = 32
    
    jobs = []
    for job_id, lr in enumerate(learning_rates):
        job_dir = f"jobs/exp2/job{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        
        # Copy all python files to job directory
        for py_file in glob.glob("*.py"):
            shutil.copy2(py_file, job_dir)
        
        cmd = [
            "cd", job_dir, "&&",
            "srun",
            *[f"{k}={v}" if v else k for k,v in slurm_args.items()],
            "python", "main_job.py",
            "--learning_rate", str(lr),
            "--num_epochs", "10", 
            "--reasoning_steps", str(reasoning_steps),
            "--lora_dim", str(lora_dim)
        ]
        
        process = subprocess.Popen(" ".join(cmd), shell=True)
        jobs.append((process, job_dir, {
            "job_id": job_id,
            "reasoning_steps": reasoning_steps,
            "learning_rate": lr,
            "lora_dim": lora_dim
        }))
    
    return jobs

def run_experiment_3(slurm_args):
    """Sweep over LoRA dimensions with fixed reasoning steps and learning rate"""
    setup_job_directory("exp3")
    
    lora_dims = [8, 16, 32, 64]
    reasoning_steps = 30
    learning_rate = 1e-3
    
    jobs = []
    for job_id, dim in enumerate(lora_dims):
        job_dir = f"jobs/exp3/job{job_id}"
        os.makedirs(job_dir, exist_ok=True)
        
        # Copy all python files to job directory
        for py_file in glob.glob("*.py"):
            shutil.copy2(py_file, job_dir)
        
        cmd = [
            "cd", job_dir, "&&",
            "srun",
            *[f"{k}={v}" if v else k for k,v in slurm_args.items()],
            "python", "main_job.py",
            "--learning_rate", str(learning_rate),
            "--num_epochs", "10",
            "--reasoning_steps", str(reasoning_steps),
            "--lora_dim", str(dim)
        ]
        
        process = subprocess.Popen(" ".join(cmd), shell=True)
        jobs.append((process, job_dir, {
            "job_id": job_id,
            "reasoning_steps": reasoning_steps,
            "learning_rate": learning_rate,
            "lora_dim": dim
        }))
    
    return jobs

def gather_results(jobs):
    """Wait for all jobs to complete and gather results"""
    results = []
    
    # Wait for all processes to complete
    for process, job_dir, params in jobs:
        process.wait()
        
        # Read results.json
        try:
            with open(os.path.join(job_dir, "results.json"), "r") as f:
                job_results = json.load(f)
                results.append({
                    **params,
                    "accuracy": job_results["accuracy"]
                })
        except FileNotFoundError:
            print(f"Warning: No results found in {job_dir}")
    
    # Sort results by job_id
    results.sort(key=lambda x: x["job_id"])
    return results

def main():
    # Create jobs directory
    os.makedirs("jobs", exist_ok=True)
    
    # Slurm arguments
    slurm_args = {
        "--account": "gts-jromberg3",
        "--gpus": "1",
        "--time": "24:00:00",
        "--mem": "16G",
        "--cpus-per-task": "4"
    }
    
    # Run all experiments
    all_jobs = []
    # all_jobs.extend(run_experiment_1(slurm_args))
    all_jobs.extend(run_experiment_2(slurm_args)) 
    # all_jobs.extend(run_experiment_3(slurm_args))
    
    # Gather and save results
    results = gather_results(all_jobs)
    
    with open("jobs/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
