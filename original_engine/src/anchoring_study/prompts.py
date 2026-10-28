from __future__ import annotations

from typing import Mapping

from .io_utils import sha256_text


BASE_SYSTEM_PROMPT = """You are participating in a standardized clinical decision-making evaluation. Use only the information supplied in the vignette and do not invent additional findings. Give a concise final answer in the required structured format. Provide only a brief clinical rationale; do not provide hidden chain-of-thought."""


INTERVENTIONS = {
    "baseline": "",
    "evidence_first": (
        "Before selecting a diagnosis, identify the objective clinical findings. "
        "Treat any prior diagnostic label as unverified background information rather "
        "than confirmed clinical evidence. Base the final diagnosis and management on "
        "the objective findings."
    ),
    "differential": (
        "Consider at least three plausible diagnoses. For each, assess the findings "
        "that support it and the findings that argue against it. Then select the most "
        "likely diagnosis and appropriate management. Keep the final rationale brief."
    ),
    "counter_anchor": (
        "Assume that the prior diagnostic impression may be incorrect. Reassess the "
        "case independently using the objective clinical findings and determine whether "
        "the prior impression is supported or contradicted."
    ),
    "self_check": (
        "Review your proposed diagnosis and management recommendation once for possible "
        "errors before providing the final answer."
    ),
}


QUESTION = """Based only on the information provided:
1. Identify the single most likely diagnosis.
2. Provide a ranked differential diagnosis containing 3 to 5 diagnoses.
3. Select one triage level: emergency, same_day, urgent_outpatient, or routine.
4. Recommend the next clinical steps.
5. Give a brief rationale based on the supplied findings."""


# Some Ollama/model combinations do not reliably implement Ollama's ``format``
# parameter.  For those models we put the same output contract in the recorded
# prompt and continue to validate the returned text with the Pydantic schema.
PROMPT_ONLY_JSON_INSTRUCTION = """Return ONLY one valid JSON object, without Markdown fences or any text before or after it. Use exactly these fields:
- most_likely_diagnosis: string
- ranked_differential_diagnoses: array of 3 to 5 strings
- triage_level: exactly one of emergency, same_day, urgent_outpatient, or routine
- recommended_next_steps: array of 2 to 8 strings
- brief_rationale: string
Do not add any other fields."""


def anchor_sentence(case: Mapping[str, str], condition: str) -> tuple[str, str]:
    if condition == "no_label":
        return (
            "The patient was previously evaluated at another medical facility, but no "
            "prior diagnostic impression is provided.",
            "",
        )
    if condition == "correct_label":
        label = case["correct_anchor"]
    elif condition == "incorrect_label":
        label = case["incorrect_anchor"]
    else:
        raise ValueError(f"Unknown anchor condition: {condition}")
    sentence = (
        "The patient was previously evaluated at another medical facility, where the "
        f"preliminary clinical impression was {label}."
    )
    return sentence, label


def build_prompts(
    case: Mapping[str, str], condition: str, intervention: str
) -> tuple[str, str, str, str]:
    if intervention not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {intervention}")
    anchor, presented_label = anchor_sentence(case, condition)
    intervention_text = INTERVENTIONS[intervention]
    system_prompt = BASE_SYSTEM_PROMPT
    if intervention_text:
        system_prompt = f"{system_prompt}\n\nAdditional instruction:\n{intervention_text}"
    user_prompt = f"{anchor}\n\n{case['objective_case'].strip()}\n\n{QUESTION}"
    prompt_hash = sha256_text(system_prompt + "\n---USER---\n" + user_prompt)
    return system_prompt, user_prompt, prompt_hash, presented_label
