import os
import subprocess

current_dir = os.path.abspath(".")

INPUT_FILE_FOLDER = os.path.join(current_dir, "test_files")
FOLDER_TO_RUN = {
    "sequential": os.path.join(current_dir, "sequential/bin/kmeans"),
    "cuda": os.path.join(current_dir, "cuda/bin/kmeans"),
    "mpi_openmp": os.path.join(current_dir, "mpi_openmp/bin/kmeans")
}
LOG_DIR = os.path.join(current_dir, "logs")

NUM_CLUSTERS = [3]
CHANGES = 0
THRESHOLD = 0
ITERATIONS = 300

def build_executables():
    """Runs 'make' in each folder to ensure the executables are compiled."""
    for model, _ in FOLDER_TO_RUN.items():
        print(f"Building {model}")
        subprocess.run(["make"], cwd=model, check=True)

def generate_and_submit_jobs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for model, executable in FOLDER_TO_RUN.items():
        folder_results = os.path.join("results", os.path.basename(model))
        os.makedirs(folder_results, exist_ok=True)

        for input_file in os.listdir(INPUT_FILE_FOLDER):
            input_path = os.path.join(INPUT_FILE_FOLDER, input_file)
            
            for cluster in NUM_CLUSTERS:
                # Define result output files
                output_file = os.path.join(current_dir, f"results/{model}/{input_file}_clusters_{cluster}.out")
                log_file = os.path.join(current_dir, f"results/{model}/{input_file}_clusters_{cluster}.log")

                # Condor submit file path
                condor_file = f"{model}_{input_file}_clusters_{cluster}.sub"
                
                # Create condor submission script based on model type
                with open(condor_file, "w") as f:
                    if model == "sequential":
                        f.write(f"""universe = vanilla
                                log = {LOG_DIR}/{model}_job.log
                                output = {LOG_DIR}/{model}_job.out
                                error = {LOG_DIR}/{model}_job.err
                                executable = {executable}
                                arguments = {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {output_file} {log_file}
                                getenv = True
                                queue
                                """)
                    elif model == "cuda":
                        f.write(f"""universe = vanilla
                                log = logs/{model}_job.log
                                output = logs/{model}_job.out
                                error = logs/{model}_job.err
                                request_gpus = 1
                                executable = {executable}
                                arguments = {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {output_file} {log_file}
                                getenv = True
                                queue
                                """)
                    elif model == "mpi_openmp":
                        f.write(f"""universe = parallel
                                executable = ./mpi_openmp/openmpiscript.sh
                                arguments = {executable} {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {output_file} {log_file}
                                should_transfer_files = YES
                                transfer_input_files = {executable}
                                when_to_transfer_output = on_exit_or_evict
                                output = logs/out.$(NODE)
                                error = logs/err.$(NODE)
                                log = logs/log
                                machine_count = 4
                                request_cpus = 8
                                getenv = True
                                queue
                                """)

                # Submit the job
                print(f"Submitting {condor_file} for {model} (clusters={cluster}, input={input_file})")
                subprocess.run(["condor_submit", condor_file], check=True)

if __name__ == "__main__":
    build_executables()
    generate_and_submit_jobs()
