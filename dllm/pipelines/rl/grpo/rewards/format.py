"""Format-related reward functions (XML tags, reasoning/answer structure)."""

import re

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_xml_answer(text: str) -> str:
    """Strict <answer>...</answer> extractor. Returns the whole input if the
    opening tag is absent (legacy behaviour). Used by int/format rewards that
    must stay sensitive to the XML structure."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_answer_lenient(text: str) -> str | None:
    """Robust answer extractor for correctness checking.

    Tries, in order:
      1. <answer>...</answer>    (requested format)
      2. #### <n>                (GSM8K native format, also LLaDA's natural
                                  output when it drops the XML wrapper)
      3. \\boxed{<n>}            (math-style)
      4. Last number in the completion (very permissive fallback)

    Returns the normalised numeric string (commas stripped) or None if no
    number could be found at all.
    """
    if not text:
        return None

    # (1) Explicit <answer> tag only counts if the tag is actually present,
    # otherwise split() returns the whole completion (== all reasoning text).
    if "<answer>" in text:
        inner = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
        nums = _NUMBER_RE.findall(inner)
        if nums:
            return nums[-1].replace(",", "")

    # (2) GSM8K-native "#### <answer>" marker.
    if "####" in text:
        tail = text.split("####", 1)[1]
        nums = _NUMBER_RE.findall(tail)
        if nums:
            return nums[0].replace(",", "")

    # (3) \boxed{...}
    m = re.search(r"\\boxed\{([^{}]*)\}", text)
    if m:
        nums = _NUMBER_RE.findall(m.group(1))
        if nums:
            return nums[-1].replace(",", "")

    # (4) Last number anywhere (most permissive).
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].replace(",", "")

    return None


def count_xml(text) -> float:
    """XML structure reward.

    Originally this function also subtracted a small penalty for any text
    appearing AFTER the closing </answer> tag (`count -= len(tail)*0.001`).
    In practice LLaDA fills a fixed-length window and keeps generating
    garbage after </answer>, so those penalties routinely pushed the mean
    of this reward NEGATIVE and counter-acted the correctness signal on
    completions that *did* close the answer tag correctly (see run11
    diagnosis in docs/RUN_HISTORY.md). The penalty was removed in run12.
    """
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
    if text.count("\n</answer>") == 1:
        count += 0.125
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def reward_len(completions, **kwargs):
    return [-len(completion[0]["content"]) for completion in completions]
