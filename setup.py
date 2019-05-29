import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# Example
if __name__ == '__main__':
    with open('requirements.txt', 'r') as r:
        for pack in r:
            install(pack)