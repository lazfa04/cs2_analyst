"""Phase 4: round win probability, and the round swing rating built on it.

features.py  each round as a sequence of game states (after every kill, the plant, the defuse)
model.py     XGBoost on those states, evaluated leave-one-demo-out against simple baselines
swing.py     how much each kill, plant and defuse moved the win probability, credited to players
"""
