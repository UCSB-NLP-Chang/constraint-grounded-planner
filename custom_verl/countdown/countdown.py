import re
import hashlib
from typing import Dict, Tuple, Optional
import subprocess
import json
import time
import random
import signal
import sys
import os
import concurrent.futures
import numpy as np

from custom_verl.reward_utils import RewardType


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    solution_str = solution_str.split("\n")[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception:
        return None


# NOTE: Modified based on https://github.com/Jiayi-Pan/TinyZero/blob/main/verl/utils/reward_score/countdown.py
# We include mechanism to control FPR and FNR rate
def compute_score(
    solution_str: str,
    ground_truth: dict,
    format_reward: int = 0.1,
    answer_reward: float = 1.0,
    FPR: float = 0.0,  # False Positive Rate (default: 0)
    FNR: float = 0.0,  # False Negative Rate (default: 0)
    **kwargs,
) -> float:
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]
    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None or equation == "":
        if do_print:
            print("No equation found")
        return {"reward": 0, "reward_type": RewardType.FormatError}

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print("Invalid equation")
        return {"reward": format_reward, "reward_type": RewardType.CompileError}

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print("Could not evaluate equation")
            return {"reward": format_reward, "reward_type": RewardType.ExecutionError}

        is_correct = abs(result - target) < 1e-5  # Account for floating point precision

        # Introduce randomness based on FPR and FNR
        if is_correct:
            if (
                random.random() < FNR
            ):  # False Negative: wrongly mark correct as incorrect
                if do_print:
                    print(f"False Negative: equation = {result}, target = {target}")
                return {"reward": format_reward, "reward_type": RewardType.FalseWrong}
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return {"reward": answer_reward, "reward_type": RewardType.Correct}
        else:
            if (
                random.random() < FPR
            ):  # False Positive: wrongly mark incorrect as correct
                if do_print:
                    print(f"False Positive: equation = {result}, target = {target}")
                return {"reward": answer_reward, "reward_type": RewardType.FalseCorrect}
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            # return format_reward, RewardType.Wrong
            return {"reward": format_reward, "reward_type": RewardType.Correct}
    except:
        if do_print:
            print("Error evaluating equation")
        # return format_reward, RewardType.ExecutionError
        return {"reward": format_reward, "reward_type": RewardType.ExecutionError}
