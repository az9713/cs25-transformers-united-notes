"""CS25 cumulative finite checks for reviewed reading chapters. Standard library only.
No GPU, neural benchmark or course completion is certified.
"""
from pathlib import Path
from runpy import run_path

REVIEWED_CHAPTERS = (1, 2, 3, 4, 6)

if __name__ == '__main__':
    for chapter in REVIEWED_CHAPTERS:
        print(f'Chapter {chapter}:')
        run_path(str(Path(__file__).with_name(f'chapter{chapter:02d}.py')), run_name='__main__')
    print(f'All {len(REVIEWED_CHAPTERS)} reviewed chapter programs passed their finite checks.')
