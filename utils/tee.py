import re
import sys
from pathlib import Path

class Tee(object):
    """Class to redirect output to both stdout and file"""
    def __init__(self, filename, verbose):
        Path(filename).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filename, "w")
        self.verbose = verbose
        if self.verbose:
            self.stdout = sys.stdout

    def write(self, message):
        if self.verbose:
            self.stdout.write(message)
        # replace `\r` and `033\[K` which are nice in the console, but we don't want those in the log file
        message = re.sub(r"(\r|\033\[K)", "\n", message)
        self.file.write(message)

    def flush(self):
        self.file.flush()
        if self.verbose:
            self.stdout.flush()