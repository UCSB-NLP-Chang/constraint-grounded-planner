# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# from . import gsm8k, math, prime_math, prime_code

import os


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    configs=None,
    no_format_score=False,
):
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif "box" in data_source:
        from custom_verl.robotic import box_reward

        USE_BOX_EFFICIENCY = configs.get("box_efficiency", False)
        USE_NOTHINKING = configs.get("box_nothinking", False)
        if USE_BOX_EFFICIENCY:
            res = box_reward.compute_score_with_step_penalty(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
        elif USE_NOTHINKING:
            res = box_reward.compute_score_nothinking(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
        else:
            res = box_reward.compute_score(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source.startswith("countdown"):
        from custom_verl.countdown import countdown

        if configs is not None:
            FPR = configs.get("countdown_FPR", 0.0)
            FNR = configs.get("countdown_FNR", 0.0)
            print("###### We manipulate FPR and FNR rate for the verifier ######")
        else:
            FPR = 0.0
            FNR = 0.0

        res = countdown.compute_score(
            solution_str,
            ground_truth,
            FPR=FPR,
            FNR=FNR,
            format_reward=0 if no_format_score else 0.1,
        )
    elif data_source.startswith("math") or data_source.startswith("AIME"):
        if configs is not None:
            math_func = configs.get("math_func", "default")
        else:
            math_func = "default"

        if math_func == "default":
            from . import math

            res = math.compute_score(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
        elif math_func == "qwen":
            from custom_verl.mathjudge import qwen_math

            #! For this function, we acceleate it by using multiprocessing and custom_naive.py
            res = qwen_math.compute_score(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
            return res
        elif math_func == "mathverify":
            from custom_verl.mathjudge import math_verify

            res = math_verify.compute_score(
                solution_str, ground_truth, format_score=0 if no_format_score else 0.1
            )
        else:
            raise ValueError(f"Unknown math_func: {math_func}")

    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0]), res[1]


def compute_score_dict(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    configs=None,
    no_format_score=False,
):
    if "box3d" in data_source:
        from custom_verl.robotic import box_reward

        res = box_reward.compute_score_with_step_penalty_3d(
            solution_str,
            ground_truth,
            # format_score=0 if no_format_score else 0.1,
            format_score=0.1,
            return_dict=True,
        )
    elif "box" in data_source:
        from custom_verl.robotic import box_reward

        USE_BOX_EFFICIENCY = configs.get("box_efficiency", False)
        USE_NOTHINKING = configs.get("box_nothinking", False)
        if USE_BOX_EFFICIENCY:
            res = box_reward.compute_score_with_step_penalty(
                solution_str,
                ground_truth,
                format_score=0 if no_format_score else 0.1,
                return_dict=True,
            )
        elif USE_NOTHINKING:
            res = box_reward.compute_score_nothinking(
                solution_str,
                ground_truth,
                format_score=0 if no_format_score else 0.1,
                return_dict=True,
            )
        else:
            res = box_reward.compute_score(
                solution_str,
                ground_truth,
                format_score=0 if no_format_score else 0.1,
                return_dict=True,
            )
    elif "countdown" in data_source:
        from custom_verl.countdown import countdown

        if configs is not None:
            FPR = configs.get("countdown_FPR", 0.0)
            FNR = configs.get("countdown_FNR", 0.0)
            print("###### We manipulate FPR and FNR rate for the verifier ######")
        else:
            FPR = 0.0
            FNR = 0.0

        res = countdown.compute_score(
            solution_str,
            ground_truth,
            FPR=FPR,
            FNR=FNR,
            format_reward=0 if no_format_score else 0.1,
            return_dict=True,
        )
    else:
        raise NotImplementedError("Unkown data source: {}".format(data_source))

    return res
