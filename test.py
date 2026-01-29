import sys
import subprocess

# Install kagglehub using subprocess - capture output to see errors
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "kagglehub"],
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
except Exception as e:
    print(f"Installation error: {e}")

# Try importing kagglehub anyway (it might already be installed)
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)