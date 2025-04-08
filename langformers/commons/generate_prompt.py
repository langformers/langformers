from typing import Dict
from langformers.commons import data_labeller_prompt_system, multi_label_prompt_allowed


def generate_prompt(text: str, conditions: Dict[str, str], multi_label: bool = False):
    """
    Generates a structured prompt for the text generation model.

    Args:
        text (str): The input text to be labeled.
        conditions (Dict[str, str]): A dictionary where keys are labels and
            values are descriptions of each label.
        multi_label (bool, default=False): If True, allows multiple labels to be selected.

    Notes:
        - If `multi_label` is enabled, the prompt includes instructions
          to allow multiple labels separated by commas.
        - The model is instructed to return only the label(s) without explanations.
    """
    system_prompt = data_labeller_prompt_system
    conditions_str = "\n".join([f"- {label}: {condition}" for label, condition in conditions.items()])
    user_prompt = (f"Here is a text for you to label.\n"
                   f"\nText: \"{text}\"\n\nConditions:\n{conditions_str}\n\n"
                   "Respond with only the label (and nothing else) that best fits the text, without explanation. ")
    multi_label_prompt = multi_label_prompt_allowed

    return [{'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt if not multi_label else user_prompt + multi_label_prompt}]
