"""Set each model only after the team's evaluation and final selection.

None means explicit demonstration mode. A selected but unregistered model
returns an unavailable error, never a demo result. Restart after changing this.
"""
FINAL_MODELS = {"species": None, "freshness": None}
