"""Reproducible cost model for ICCE 2026 Paper 206."""
# Break-even vs API: training_cost / (api_per_dialogue - local_inf_per_dialogue)
def break_even(training_cost: float, api: float, local: float = 0.0) -> float:
    return training_cost / (api - local)

assert abs(break_even(45, 0.045) - 1000) < 1e-9
assert abs(break_even(100, 0.045) - 2222.222222222222) < 1e-6
print("C2 BE vs API:", break_even(45, 0.045))
print("C3 BE vs API ($100):", round(break_even(100, 0.045)))
print("C3 BE vs API ($55):", round(break_even(55, 0.045)))
