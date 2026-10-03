"""Original finite checks for all nine independently reviewed CS25 chapters.
Standard library only; no neural/GPU benchmark or course certification.
"""
from pathlib import Path
from runpy import run_path

REVIEWED_CHAPTERS = tuple(range(1, 10))

if __name__ == '__main__':
    for chapter in REVIEWED_CHAPTERS:
        print(f'Chapter {chapter}:')
        run_path(str(Path(__file__).with_name(f'chapter{chapter:02d}.py')), run_name='__main__')
    print(f'All {len(REVIEWED_CHAPTERS)} reviewed chapter programs passed their finite checks.')
