import multiprocessing
from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig

from custom_verl.reward_utils import RewardType


def compute_score_process(answer, ground_truth, output_queue):
    try:
        result = verify(ground_truth, answer)
    except:
        result = False
    finally:
        output_queue.put(result)


def compute_score(solution_str, ground_truth, format_score=0.1) -> float:
    # CPU-bound synchronous work
    if not ground_truth.startswith("$"):
        ground_truth = f"${ground_truth}$"
    gold = parse(
        ground_truth,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    answer = parse(
        solution_str,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    if len(answer) == 0:
        return 0, RewardType.FormatError
    else:
        acc = verify(gold, answer)
        score = format_score if (not acc) else acc
        return score, RewardType.FormatCorrect if (not acc) else RewardType.Correct
