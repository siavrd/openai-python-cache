import json
import time
from typing import Optional

from openai import OpenAI, APIError
from openai.types.chat.chat_completion import ChatCompletion as ChatCompletionType
from .provider import Sqlite3CacheProvider


class CachedChatCompletion:
    """Wrapper around OpenAI Chat Completions with optional SQLite-based caching.

    This class wraps the OpenAI Chat Completions API and adds transparent caching
    using a `Sqlite3CacheProvider`. Cached responses are stored and automatically
    reused for identical requests to reduce API usage and latency.

    It is compatible with the modern OpenAI Python SDK (>=1.0.0).

    Example:
        >>> from openai_python_cache.api import CachedChatCompletion
        >>> from openai_python_cache.provider import Sqlite3CacheProvider
        >>> cache = Sqlite3CacheProvider("cache.sqlite")
        >>> chat = CachedChatCompletion(api_key="sk-...")
        >>> response = chat.create(
        ...     model="gpt-4o-mini",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     cache_provider=cache,
        ... )
        >>> print(response.choices[0].message.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Optional[OpenAI] = None,
    ):
        """Initialize a CachedChatCompletion instance.

        Args:
            api_key (Optional[str]): The OpenAI API key. If None and no client is provided,
                the key will be automatically loaded from environment variables.
            client (Optional[OpenAI]): An existing OpenAI client instance. If provided,
                it will be used instead of creating a new one.
        """
        if client is not None:
            # Use the provided client instance (for reuse or custom configuration)
            self.client = client
        else:
            # Create a new OpenAI client using the provided API key
            self.client = OpenAI(api_key=api_key)

    def create(
        self,
        cache_provider: Optional[Sqlite3CacheProvider] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionType:
        """Create a chat completion with optional caching and retry logic.

        If a cache provider is supplied, this method checks whether the same request
        has already been made and reuses the cached response when available.

        Args:
            cache_provider (Optional[Sqlite3CacheProvider]): The cache provider instance.
                If provided, responses are cached and reused for identical parameters.
            timeout (Optional[float]): Maximum time in seconds to retry the request
                if the model is warming up or temporarily unavailable.
            **kwargs: Parameters passed directly to `client.chat.completions.create()`,
                such as `model`, `messages`, `temperature`, etc.

        Returns:
            ChatCompletionType: The structured API response object from OpenAI.
        """
        start = time.time()

        # If no cache provider is provided, just perform a direct API call
        if cache_provider is None:
            return self.client.chat.completions.create(**kwargs)

        # Generate a deterministic cache key from request parameters
        cache_key = cache_provider.hash_params(kwargs)

        # Try to load a response from cache
        cached_response = cache_provider.get(cache_key)
        if cached_response:
            # Convert JSON string -> dict -> ChatCompletionType object
            cached_dict = json.loads(cached_response)
            return ChatCompletionType.model_validate(cached_dict)

        # Cache miss → perform the API request and store the response
        while True:
            try:
                # Call the OpenAI Chat Completions API
                response = self.client.chat.completions.create(**kwargs)

                # Store the response in cache as a JSON-serializable dict
                cache_provider.insert(cache_key, kwargs, response.model_dump())

                # Return the live API response
                return response

            except APIError as e:
                # Handle transient API errors (e.g., model warm-up or overload)
                if timeout is not None and time.time() > start + timeout:
                    # Timeout reached → re-raise the exception
                    raise

                # Log the issue and retry after a short delay
                print(f"[Info] Model not ready, retrying... ({e})")
                time.sleep(1)
