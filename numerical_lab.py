"""CS25 cumulative finite checks, beginning with the overview pilot."""
from pathlib import Path
from runpy import run_path

if __name__ == '__main__':
    run_path(str(Path(__file__).with_name('chapter01.py')), run_name='__main__')
