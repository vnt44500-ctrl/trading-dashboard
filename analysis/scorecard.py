"""Long-horizon, time-consistent scorecard for the signal engine.

See also: recommendations-agreed-and-approved.json (the accepted attack plan),
analysis/recommendations_agreed_and_approved.py (the same plan as runnable references),
and analysis/recommendations-agreed-and-approved.md (the same plan as readable notes)."""


import json
import os
import sys

ws = r"C:\Users\visha\trading-dashboard"
if ws not in sys.path:
    sys.path.insert(0, ws)

from analysis import signal_models as sm
from analysis import panel_models

PATH = os.path.join(os.path.dirname(__file__), "recommendations_agreed_and_approved.json")
if os.path.exists(PATH):
    with open(PATH, encoding="utf-8") as f:
        RESOURCE = json.load(f)