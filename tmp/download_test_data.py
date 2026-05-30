import os
import sys

def download_data():
    try:
        import gdown
    except ImportError:
        print("gdown package is not installed. Installing it now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    os.makedirs('data', exist_ok=True)
    url = 'https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V'
    output = 'data/test.csv'
    
    print(f"Downloading test.csv from Google Drive to {output}...")
    gdown.download(url, output, quiet=False)
    print("Download completed successfully!")

if __name__ == '__main__':
    download_data()
