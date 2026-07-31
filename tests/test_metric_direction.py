"""Direction tests for SLR and composite."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.metric_slr import compute_slr
from src.evaluation.composite_score import compute_composite, DEFAULT_WEIGHTS

def test_direct_answer_leaks_more_than_question():
    leak = {"turns": [{"role": "tutor", "content": "The answer is x = 4."},
                      {"role": "student", "content": "ok"}]}
    ask = {"turns": [{"role": "tutor", "content": "What do you notice about both sides?"},
                     {"role": "student", "content": "they look similar"}]}
    # higher score = less leaking
    assert compute_slr(leak) < compute_slr(ask)

def test_weights_sum_to_one():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9

if __name__ == "__main__":
    test_direct_answer_leaks_more_than_question()
    test_weights_sum_to_one()
    print("OK")
