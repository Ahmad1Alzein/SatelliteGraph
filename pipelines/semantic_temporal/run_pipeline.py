"""Standalone access to the same native general pipeline CLI."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.cli import main

if __name__=='__main__':
    arguments=sys.argv[1:]
    if not arguments or arguments[0] not in ('run','segment','graph','change','--help','-h'): arguments=['run',*arguments]
    main(['pipeline',*arguments])
