import os
import sys
import subprocess

def main():
    env = os.environ.copy()
    python_executable = sys.executable

    # Run the main function of findpeaksforproject.py
    result1 = subprocess.run([python_executable, 'findpeaksforstandardall.py'], env=env)

    if result1.returncode == 0:
        # Run the main function of clusterforPeaksProject.py
        result2 = subprocess.run([python_executable, 'clusterforPeaksStandardAll.py'], env=env)

if __name__ == '__main__':
    main()
