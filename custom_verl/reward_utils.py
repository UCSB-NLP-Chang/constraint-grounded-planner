from enum import Enum


# We list all possible outcomes for a reward function
class RewardType(Enum):
    # Reward types
    FormatError = 0
    FormatCorrect = 1
    CompileError = 2
    ExecutionError = 3
    TimeoutError = 4
    Correct = 5
    Wrong = 6
    FalseCorrect = 7
    FalseWrong = 8
    EarlyStopped = 9  # this run case does not return because others early exit

    # TestCaseJudge specific reward types
    PassGoodFailBad = 10  # correct testcase
    PassGoodPassBad = 11  # incorrect testcase
    FailGood = 12  # incorrect testcase

    # RobotEnv specific reward types
    CollisionObject = 13
    CollisionRobot = 14
    InvalidAction = 15
    ParseError = 16
    StepParseError = 17
    StepSuccess = 18
