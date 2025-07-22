import os


def compute_score_dict(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    configs=None,
    no_format_score=False,
):
    if "box" in data_source:
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
    else:
        raise NotImplementedError("Unkown data source: {}".format(data_source))

    return res
