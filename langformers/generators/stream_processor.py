import requests
import json


def generate_chunks(response, key_name, encoding):
    """Yields chunks from the SSE stream."""
    for line in response.iter_lines():
        if line:
            try:
                decoded_line = line.decode(encoding)
                if decoded_line.startswith("data:"):
                    try:
                        data = json.loads(decoded_line[5:].strip())
                        if key_name in data and data[key_name]:
                            yield data[key_name]
                    except json.JSONDecodeError:
                        continue
            except UnicodeDecodeError:
                continue


def get_full_response(response, key_name, encoding):
    """Accumulates and returns the full response."""
    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                decoded_line = line.decode(encoding)
                if decoded_line.startswith("data:"):
                    try:
                        data = json.loads(decoded_line[5:].strip())
                        if key_name in data and data[key_name]:
                            full_response += data[key_name]
                    except json.JSONDecodeError:
                        continue
            except UnicodeDecodeError:
                continue
    return full_response


class StreamProcessor:
    """
       Handles Server-Sent Events (SSE) responses.
    """
    def __init__(self, headers):
        """
        Initializes the StreamProcessor class.

        Args:
            headers (dict): A dictionary of headers. Below is an example. Headers also contain API keys, Bearer tokens, etc.

                .. code-block:: python

                    headers = {
                        "Content-Type": "application/json"
                    }
        """
        self.headers = headers

    def process(self, endpoint_url: str, payload: dict, key_name: str = "chunk", stream: bool = True, encoding: str = "utf-8"):
        """
        Processes an API response that sends Server-Sent Events (SSE).

        Args:
            endpoint_url (str): The API endpoint (e.g., http://0.0.0.0:8000/api/generate).
            payload (dict): The request payload.
            key_name (str, default="chunk"): The custom key name used in the SSE streams.
            stream (bool, default=True): Whether to stream outputs.
            encoding (str, default="utf-8"): The encoding to use for JSON encoding.
        """
        try:
            response = requests.post(
                endpoint_url,
                headers=self.headers,
                json=payload,
                stream=stream
            )
            response.raise_for_status()

            if stream:
                return generate_chunks(response, key_name, encoding)
            else:
                return get_full_response(response, key_name, encoding)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}") from e
