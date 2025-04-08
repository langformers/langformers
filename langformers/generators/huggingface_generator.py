import asyncio
import json
import os
from threading import Thread
from typing import Optional, Callable, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from langformers.commons import default_chat_prompt_system, print_message
from langformers.commons import device as device_type  # to avoid conflicting with "device" imported variable


class HuggingFaceGenerator:
    """
    A FastAPI- and Hugging Face-powered LLM interface (user interface and REST api).
    """

    def __init__(self, model_name: str, memory: bool = True, dependency: Optional[Callable[..., Any]] = None,
                 device: str = None):
        """
        Initializes the Hugging Face chat generator with the specified model.

        Args:
            model_name (str): The name/path of the Hugging Face model to use
            memory (bool, default=True): Whether to retain conversation history
            dependency (Optional[Callable[..., Any]], default=<no auth>): A FastAPI dependency.
                The callable can return any value which will be injected into the route `/api/generate`.
            device (str, default=None): The device to load the model, data on ("cuda", "mps" or "cpu").
                If not provided, device will automatically be inferred. Particularly used for HuggingFace models, input ids and attention mask needs to be on save device as the model.
        """
        self.model_name = model_name
        self.memory = memory
        self.messages = []
        self.device = device_type if device is None else device
        self.app = FastAPI()
        self.dependency = dependency

        if self.dependency:
            print_message(f"Provided authentication dependency will be used for LLM inference.")
        else:
            print_message(f"No dependency provided. LLM inference will be done "
                          f"without authentication.")
        try:
            print_message(f"Loading Hugging Face model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

            print_message("Hugging Face model loaded: {model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name} on device `{self.device}`: {str(e)}")


        base_dir = os.path.dirname(os.path.abspath(__file__))
        templates_dir = os.path.join(base_dir, "chat")
        static_dir = os.path.join(base_dir, "static")

        self.templates = Jinja2Templates(directory=templates_dir)
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        self.setup_routes()

    async def generate(self, prompt: str, system_prompt: str = default_chat_prompt_system,
                            memory_k: int = 10, temperature: float = 0.5,
                            top_p: float = 1, max_length: int = 5000):
        """
        Generates an LLM response to a given prompt.

        Args:
            prompt (str, required): The user's input prompt.
            system_prompt (str, default=default_chat_prompt_system): The system-level instruction for the LLM.
            memory_k (int, default=10): The number of previous messages to retain in memory.
            temperature (float, default=0.5): Controls randomness of responses (higher = more random).
            top_p (float, default=1): Nucleus sampling parameter (lower = more focused).
            max_length (int, default=5000): Maximum number of tokens to generate.

        Notes:
            - Maintains conversation memory if enabled.
            - Streams responses in real time.
        """
        if not prompt or not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="Invalid 'prompt'. It must be a non-empty string.")
        if not (0 <= temperature <= 1):
            raise HTTPException(status_code=400, detail="'temperature' must be between 0 and 1.")
        if not (0 <= top_p <= 1):
            raise HTTPException(status_code=400, detail="'top_p' must be between 0 and 1.")
        if max_length <= 0:
            raise HTTPException(status_code=400, detail="'max_length' must be a positive integer.")

        user_prompt = {"role": "user", "content": prompt}
        system_prompt = {'role': 'system', 'content': system_prompt}

        if self.memory:
            if not self.messages or self.messages[0] != system_prompt:
                self.messages = [system_prompt]
            self.messages.append(user_prompt)
            if len(self.messages) > memory_k:
                self.messages = [system_prompt] + self.messages[-memory_k:]
            messages = self.messages
        else:
            messages = [system_prompt, user_prompt]

        try:
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error formatting text: {str(e)}")


        try:
            tokenized_input = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                return_attention_mask=True
            ).to(self.device)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error tokenizing input: {str(e)}")


        try:
            input_ids = tokenized_input.input_ids
            attention_mask = tokenized_input.attention_mask

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                timeout=20,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                streamer=streamer,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True if temperature > 0 else False
            )

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            async def generate():
                full_response = []
                for new_text in streamer:
                    full_response.append(new_text)
                    yield f"data: {json.dumps({'chunk': new_text})}\n\n"
                    await asyncio.sleep(0)

                if self.memory:
                    self.messages.append({
                        "role": "assistant",
                        "content": "".join(full_response)
                    })

            return StreamingResponse(generate(), media_type="text/event-stream")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during text generation: {str(e)}")

    def setup_routes(self):
        """
        Defines the FastAPI routes for handling chat interactions.

        Routes:
            - GET `/` : Renders the chat interface.
            - POST `/api/generate` : Accepts user input and returns AI-generated responses.

        Notes:
            - Uses Jinja2 templates for rendering the chat interface.
            - The API expects a JSON payload containing a `prompt` key.
        """

        @self.app.head("/")
        async def head_root():
            return Response(status_code=200)

        @self.app.get("/")
        async def root(request: Request):
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request,
                 "model_name": self.model_name,
                 "default_chat_prompt_system": default_chat_prompt_system}
            )

        @self.app.post("/api/generate")
        async def generate_text(request: Request, _: bool = Depends(self.dependency) if self.dependency else Depends(no_dependency)):
            try:
                data = await request.json()
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format.")

            if not data or 'prompt' not in data:
                raise HTTPException(status_code=400, detail="Missing 'prompt' in request body.")
            
            try:
                return await self.generate(
                    prompt=data['prompt'],
                    system_prompt=data.get('system_prompt', default_chat_prompt_system),
                    memory_k=data.get('memory_k', 10),
                    temperature=data.get('temperature', 0.5),
                    top_p=data.get('top_p', 1),
                    max_length=data.get('max_length', 5000)
                )
            except HTTPException as e:
                raise e
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Starts the FastAPI web server.

        Args:
            host (str, default="0.0.0.0"): The IP address to bind the server to.
            port (int, default=8000): The port number to listen on.
        """
        uvicorn.run(self.app, host=host, port=port)


async def no_dependency():
    """Default dependency for `/api/generate` endpoint."""
    return True
