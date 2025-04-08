from typing import List, Dict
from transformers import pipeline
from langformers.commons import generate_prompt, print_message


class HuggingFaceDataLabeller:
    """
    Performs data labelling task using generative LLMs available on Hugging Face.

    This class uses the selected generative LLM to assign labels to input text based on user-defined conditions.
    It supports both single-label and multi-label classification.
    """

    def __init__(self, model_name: str, multi_label: bool = False):
        """
        Initializes the HuggingFaceDataLabeler with the specified model.

        Args:
            model_name (str, required): The name of the Hugging Face model to use.
            multi_label (bool, default=False): If True, allows multiple labels to be selected.
        """
        self.model_name = model_name
        self.multi_label = multi_label

        try:
            self.pipe = pipeline("text-generation", model=model_name)
            print_message(f"Hugging Face model is initialized for data labelling: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the Hugging Face pipeline with model '{model_name}': {e}")

    def pipeline(self, complete_prompt: List[Dict[str, str]]):
        """
        Runs the generated prompt through the Hugging Face text generation pipeline.

        Args:
            complete_prompt (List[Dict[str, str]], required): The structured prompt to be processed.
        """
        try:
            output = self.pipe(complete_prompt, pad_token_id=self.pipe.tokenizer.eos_token_id)
        except Exception as e:
            raise RuntimeError(f"Failed to process the prompt through Hugging Face pipeline: {e}")

        return output

    def label(self, text: str, conditions: Dict[str, str]):
        """
        Labels a given text based on user-defined conditions.

        Args:
            text (str, required): The text to be classified.
            conditions (Dict[str, str], required): A dictionary mapping labels to their descriptions.
        """
        try:
            complete_prompt = generate_prompt(text, conditions, self.multi_label)
        except Exception as e:
            raise RuntimeError(f"Failed to generate prompt: {e}")

        try:
            output = self.pipeline(complete_prompt)[-1]["generated_text"][-1]["content"]
            return output.strip().lower()
        except KeyError as e:
            raise RuntimeError(f"Unexpected response format from the pipeline: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while labeling the text: {e}")
