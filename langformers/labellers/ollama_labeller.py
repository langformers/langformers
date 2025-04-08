from typing import List, Dict
from ollama import chat
from ollama import ChatResponse
from langformers.commons import generate_prompt, print_message


class OllamaDataLabeller:
    """
    Performs data labelling task using generative LLMs provided by Ollama.

    This class uses the selected generative LLM to assign labels to input text based on user-defined conditions.
    It supports both single-label and multi-label classification.
    """

    def __init__(self, model_name: str, multi_label: bool = False):
        """
        Initializes the OllamaDataLabeler with the specified model.

        Args:
            model_name (str, required): The name of the Ollama model to use.
            multi_label (bool, default=False): If True, allows multiple labels to be selected.
        """
        self.model_name = model_name
        self.multi_label = multi_label

        print_message(f"Ollama model is initialized for data labelling: {model_name}")

    def pipeline(self, complete_prompt: List[Dict[str, str]]):
        """
        Runs the generated prompt through the Ollama's `chat()`.

        Args:
            complete_prompt (List[Dict[str, str]], required): The structured prompt to be processed.
        """
        try:
            pipe: ChatResponse = chat(model=self.model_name, messages=complete_prompt)
            output = pipe.message.get('content', '')

            return output
        except Exception as e:
            raise RuntimeError(f"Failed to process the prompt through Ollama Chat(): {e}")

    def label(self, text: str, conditions: Dict[str, str]):
        """
        Labels a given text based on user-defined conditions.

        Args:
            text (str): The text to be labelled.
            conditions (Dict[str, str], required): A dictionary mapping labels to their descriptions.
        """
        try:
            complete_prompt = generate_prompt(text, conditions, self.multi_label)
        except Exception as e:
            raise RuntimeError(f"Failed to generate prompt: {e}")

        try:
            output = self.pipeline(complete_prompt)
            return output.strip().lower()
        except KeyError as e:
            raise RuntimeError(f"Unexpected response format from the pipeline: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while labeling the text: {e}")
