#!/usr/bin/env python
"""Debug test runner that dumps stack traces on hang detection."""

import subprocess
import sys
import time
import signal
import os
import threading
import traceback
from datetime import datetime

# Configuration
TIMEOUT_NO_OUTPUT = 300  # 5 minutes without output
CHECK_INTERVAL = 5  # Check every 5 seconds

def run_with_hang_detection():
    """Run pytest with hang detection and stack trace dumping."""

    # Enable faulthandler and debug logging for the subprocess
    env = os.environ.copy()
    env['PYTHONFAULTHANDLER'] = '1'

    # Start pytest process with verbose logging
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/', '-v', '--tb=short',
        '--log-cli-level=DEBUG',
        '--log-cli-format=%(asctime)s %(levelname)s %(name)s: %(message)s',
        '-p', 'no:timeout'  # Disable any pytest timeout plugins
    ]

    print(f"[{datetime.now()}] Starting test run: {' '.join(cmd)}")
    print(f"[{datetime.now()}] Will dump stack traces if no output for {TIMEOUT_NO_OUTPUT}s")
    print("-" * 80)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    last_output_time = time.time()
    output_lines = []

    def reader_thread():
        """Read output and track last output time."""
        nonlocal last_output_time
        for line in proc.stdout:
            last_output_time = time.time()
            print(line, end='', flush=True)
            output_lines.append(line)

    # Start reader thread
    reader = threading.Thread(target=reader_thread, daemon=True)
    reader.start()

    # Monitor for hangs
    try:
        while proc.poll() is None:
            time.sleep(CHECK_INTERVAL)

            elapsed_since_output = time.time() - last_output_time

            if elapsed_since_output > TIMEOUT_NO_OUTPUT:
                print("\n" + "=" * 80)
                print(f"[{datetime.now()}] HANG DETECTED!")
                print(f"No output for {elapsed_since_output:.0f} seconds")
                print("=" * 80)

                # Send SIGUSR1 to trigger faulthandler (if registered)
                # Then send SIGINT to get Python traceback
                print(f"\n[{datetime.now()}] Sending SIGINT to get traceback...")
                proc.send_signal(signal.SIGINT)
                time.sleep(5)

                if proc.poll() is None:
                    print(f"\n[{datetime.now()}] Process still running, sending SIGTERM...")
                    proc.terminate()
                    time.sleep(5)

                if proc.poll() is None:
                    print(f"\n[{datetime.now()}] Process still running, sending SIGKILL...")
                    proc.kill()

                break

    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Interrupted by user")
        proc.terminate()

    # Wait for process to finish
    reader.join(timeout=5)
    proc.wait()

    print("\n" + "=" * 80)
    print(f"[{datetime.now()}] Process exited with code: {proc.returncode}")

    return proc.returncode

if __name__ == '__main__':
    sys.exit(run_with_hang_detection())
