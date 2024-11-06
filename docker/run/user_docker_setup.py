import subprocess
import os
import uuid
import shutil

BASE_PATH = "/root/user_containers/Work_dir/"
DOCKER_IMAGE = "python:3.9-slim"  # or your preferred Docker image
GIT_REPO_URL = "https://github.com/AalindTiwari/Idkwhat.git"

def generate_container_name(user_id):
    return f"user_{user_id}_{str(uuid.uuid4())[:8]}"

def setup_user_directory(user_id):
    user_dir = os.path.join(BASE_PATH, f"user_{user_id}")
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    os.makedirs(user_dir)
    
    subprocess.run(["git", "clone", GIT_REPO_URL, user_dir])
    return user_dir

def create_docker_container(user_id):
    container_name = generate_container_name(user_id)
    user_dir = setup_user_directory(user_id)

    subprocess.run(["docker", "pull", DOCKER_IMAGE])

    subprocess.run([
        "docker", "run", "-d", "--name", container_name, "--restart", "unless-stopped",
        "-v", f"{user_dir}:/app", 
        DOCKER_IMAGE, "python", "/app/run_ui.py"  
    ])
    
    print(f"Docker container {container_name} created and running for user {user_id}")
    return container_name

def stop_container(container_name):
    subprocess.run(["docker", "stop", container_name])
    subprocess.run(["docker", "rm", container_name])
    print(f"Container {container_name} stopped and removed.")

if __name__ == "__main__":
    # Example: creating containers for users 1-15
    for user_id in range(1, 16):
        create_docker_container(user_id)
