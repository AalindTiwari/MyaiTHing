from enum import Enum
import os
from langchain_openai import (
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    AzureOpenAI,
)
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic.v1.types import SecretStr
from python.helpers.dotenv import load_dotenv
import requests  # Add this import

# environment variables
load_dotenv()

# Configuration
DEFAULT_TEMPERATURE = 0.0


class ModelType(Enum):
    CHAT = "Chat"
    EMBEDDING = "Embedding"


class ModelProvider(Enum):
    ANTHROPIC = "Anthropic"
    HUGGINGFACE = "HuggingFace"
    GOOGLE = "Google"
    GROQ = "Groq"
    LMSTUDIO = "LM Studio"
    MISTRALAI = "Mistral AI"
    NVIDIA = "NVIDIA"  # Add this line
    OLLAMA = "Ollama"
    OPENAI = "OpenAI"
    OPENAI_AZURE = "OpenAI Azure"
    OPENROUTER = "OpenRouter"
    SAMBANOVA = "Sambanova"


# Utility function to get API keys from environment variables
def get_api_key(service):
    return os.getenv(f"API_KEY_{service.upper()}") or os.getenv(
        f"{service.upper()}_API_KEY"
    )


def get_model(type: ModelType, provider: ModelProvider, name: str, **kwargs):
    fnc_name = f"get_{provider.name.lower()}_{type.name.lower()}"  # function name of model getter
    model = globals()[fnc_name](name, **kwargs)  # call function by name
    return model
# HuggingFace models


def get_huggingface_embedding(model_name: str, **kwargs):
    return HuggingFaceEmbeddings(model_name=model_name, **kwargs)

# OpenAI models
def get_openai_chat(
    model_name: str,
    api_key=get_api_key("openai"),
    temperature=DEFAULT_TEMPERATURE,
    **kwargs,
):
    return ChatOpenAI(model_name=model_name, temperature=temperature, api_key=api_key, **kwargs)  # type: ignore


def get_openai_instruct(
    model_name: str,
    api_key=get_api_key("openai"),
    temperature=DEFAULT_TEMPERATURE,
    **kwargs,
):
    return OpenAI(model=model_name, temperature=temperature, api_key=api_key, **kwargs)  # type: ignore


def get_openai_embedding(model_name: str, api_key=get_api_key("openai"), **kwargs):
    return OpenAIEmbeddings(model=model_name, api_key=api_key, **kwargs)  # type: ignore


def get_azure_openai_chat(
    deployment_name: str,
    api_key=get_api_key("openai_azure"),
    temperature=DEFAULT_TEMPERATURE,
    azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
    **kwargs,
):
    return AzureChatOpenAI(deployment_name=deployment_name, temperature=temperature, api_key=api_key, azure_endpoint=azure_endpoint, **kwargs)  # type: ignore


def get_azure_openai_instruct(
    deployment_name: str,
    api_key=get_api_key("openai_azure"),
    temperature=DEFAULT_TEMPERATURE,
    azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
    **kwargs,
):
    return AzureOpenAI(deployment_name=deployment_name, temperature=temperature, api_key=api_key, azure_endpoint=azure_endpoint, **kwargs)  # type: ignore


def get_azure_openai_embedding(
    deployment_name: str,
    api_key=get_api_key("openai_azure"),
    azure_endpoint=os.getenv("OPENAI_AZURE_ENDPOINT"),
    **kwargs,
):
    return AzureOpenAIEmbeddings(deployment_name=deployment_name, api_key=api_key, azure_endpoint=azure_endpoint, **kwargs)  # type: ignore


# Google models
def get_google_chat(
    model_name: str,
    api_key=get_api_key("google"),
    temperature=DEFAULT_TEMPERATURE,
    **kwargs,
):
    return GoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=api_key, safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}, **kwargs)  # type: ignore


# Mistral models
def get_mistral_chat(
    model_name: str,
    api_key=get_api_key("mistral"),
    temperature=DEFAULT_TEMPERATURE,
    **kwargs,
):
    return ChatMistralAI(model=model_name, temperature=temperature, api_key=api_key, **kwargs)  # type: ignore


# Groq models
def get_groq_chat(
    model_name: str,
    api_key=get_api_key("groq"),
    temperature=DEFAULT_TEMPERATURE,
    **kwargs,
):
    return ChatGroq(model_name=model_name, temperature=temperature, api_key=api_key, **kwargs)  # type: ignore


# OpenRouter models
def get_openrouter_chat(
    model_name: str,
    api_key=get_api_key("openrouter"),
    temperature=DEFAULT_TEMPERATURE,
    base_url=os.getenv("OPEN_ROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
    **kwargs,
):
    return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature, base_url=base_url, **kwargs)  # type: ignore


def get_openrouter_embedding(
    model_name: str,
    api_key=get_api_key("openrouter"),
    base_url=os.getenv("OPEN_ROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
    **kwargs,
):
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


# Sambanova models
def get_sambanova_chat(
    model_name: str,
    api_key=get_api_key("sambanova"),
    temperature=DEFAULT_TEMPERATURE,
    base_url=os.getenv("SAMBANOVA_BASE_URL") or "https://fast-api.snova.ai/v1",
    max_tokens=1024,
    **kwargs,
):
    return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature, base_url=base_url, max_tokens=max_tokens, **kwargs)  # type: ignore


# Other OpenAI compatible models
def get_other_chat(
    model_name: str,
    api_key=None,
    temperature=DEFAULT_TEMPERATURE,
    base_url=None,
    **kwargs,
):
    return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature, base_url=base_url, **kwargs)  # type: ignore


def get_other_embedding(model_name: str, api_key=None, base_url=None, **kwargs):
    return OpenAIEmbeddings(model=model_name, api_key=api_key, base_url=base_url, **kwargs)  # type: ignore


def get_jina_embedding(
    api_key=os.getenv("JINA_API_KEY"),
    model_name="jina-embeddings-v2-base-en",
    **kwargs,
):
    return JinaEmbeddings(api_key=api_key, model_name=model_name, **kwargs)

class JinaEmbeddings:
    def __init__(self, api_key, model_name="jina-embeddings-v2-base-en", **kwargs):
        if not api_key:
            raise ValueError("Jina API key is required")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://api.jina.ai/v1/embeddings"
        
    def embed_documents(self, texts):
        if not texts:
            return []
        try:
            result = self.embed_text(texts)
            # Extract just the embedding vectors from the response
            return [item["embedding"] for item in result["data"]]
        except Exception as e:
            raise ValueError(f"Error embedding documents with Jina: {str(e)}")
            
    def embed_query(self, text):
        if not text:
            return []
        try:
            result = self.embed_text([text])
            # Extract just the embedding vector from the first result
            return result["data"][0]["embedding"]
        except Exception as e:
            raise ValueError(f"Error embedding query with Jina: {str(e)}")

    def embed_text(self, texts):
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "input": texts,
            "model": self.model_name
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


# NVIDIA models
def get_nvidia_chat(
    model_name: str,
    api_key=get_api_key("nvidia"),
    temperature=DEFAULT_TEMPERATURE,
    base_url="https://integrate.api.nvidia.com/v1",
    **kwargs,
):
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )  # type: ignore

# Add this function call option
def get_nvidia_embedding(
    model_name: str,
    api_key=get_api_key("nvidia"),
    base_url="https://integrate.api.nvidia.com/v1",
    **kwargs,
):
    return OpenAIEmbeddings(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )  # type: ignore

