from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Optional
from .api import SampleParams


@dataclass
class EnvironmentConfig:
    env_name: str = "box1"
    grid_N: int = 2
    grid_M: int = 4
    num_objects: int = 2
    robot_mode: str = "full"
    robot_as: str = "agent"
    movement: str = "full"


@dataclass
class RunConfig:
    seed: int = 42
    prompt_file: str = ""
    env_name: str = "box1"
    llm_name: str = "llama"  # The name is set in the registry below
    data: str = ""
    mode: str = "one_round"
    batch_size: int = 4
    resume: bool = False
    sample_params: SampleParams = field(default_factory=SampleParams)
    env_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model_path: str = ""


cs = ConfigStore.instance()
cs.store(name="main_config", node=RunConfig)


@dataclass
class ModelRegistryParams:
    model_name: str
    raw_model_name: str
    PORT: int = None
    API_KEY: Optional[str] = field(default=None)
    API_URL: Optional[str] = field(default=None)


# List to keep track of all config objects
model_configs = []


# Function to register a model config and add it to the list
def register_model_registry(
    name: str,
    model_name: str,
    raw_model_name: str,
    port: int,
    API_KEY: str = None,
    API_URL: str = None,
):
    config = ModelRegistryParams(
        model_name=model_name,
        raw_model_name=raw_model_name,
        PORT=port,
        API_KEY=API_KEY,
        API_URL=API_URL,
    )
    cs.store(name=name, node=config)
    model_configs.append(name)


# Register all model configurations
register_model_registry(
    name="llama_registry",
    model_name="Llama-3.2-3B-Instruct@localvllm",
    raw_model_name="meta-llama/Llama-3.2-3B-Instruct",
    port=10000,
)

register_model_registry(
    name="mistral_registry",
    model_name="Mistral-7B-Instruct-v0.3@localvllm",
    raw_model_name="mistralai/Mistral-7B-Instruct-v0.3",
    port=20000,
)

register_model_registry(
    name="deepseek_registry",
    model_name="deepseek-math-7b-instruct@localvllm",
    raw_model_name="deepseek-ai/deepseek-math-7b-instruct",
    port=30000,
)

register_model_registry(
    name="qwen_registry",
    model_name="Qwen2.5-7B-Instruct@localvllm",
    raw_model_name="Qwen/Qwen2.5-7B-Instruct",
    port=40000,
)


register_model_registry(
    name="qwen-3b_registry",
    model_name="Qwen2.5-3B-Instruct@localvllm",
    raw_model_name="Qwen/Qwen2.5-3B-Instruct",
    port=40000,
)

register_model_registry(
    name="qwen3-4b_registry",
    model_name="Qwen3-4B@localvllm",
    raw_model_name="Qwen/Qwen3-4B",
    port=40000,
)

register_model_registry(
    name="qwen3-8b_registry",
    model_name="Qwen3-8B@localvllm",
    raw_model_name="Qwen/Qwen3-8B",
    port=40000,
)

register_model_registry(
    name="qwen3-14b_registry",
    model_name="Qwen3-14B@localvllm",
    raw_model_name="Qwen/Qwen3-14B",
    port=40000,
)

register_model_registry(
    name="qwen3-32b_registry",
    model_name="Qwen3-32B@localvllm",
    raw_model_name="Qwen/Qwen3-32B",
    port=40000,
)

register_model_registry(
    name="qwq_registry",
    model_name="Qwen/QwQ-32B@localvllm",
    raw_model_name="Qwen/QwQ-32B",
    port=40000,
)


register_model_registry(
    name="r1-qwen-1.5b_registry",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B@localvllm",
    raw_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    port=30000,
)


register_model_registry(
    name="my_gpt4omini_registry",
    model_name="gpt-4o-mini",
    raw_model_name="gpt-4o-mini",
    port=-1,
    API_KEY="YOUR_API_KEY",
)


register_model_registry(
    name="azure_4omini_registry",
    model_name="gpt-4o-mini@azure",
    raw_model_name="gpt-4o-mini",
    port=-1,
    API_URL="https://bairu-m5yhc9jl-westus.cognitiveservices.azure.com/",
    API_KEY="YOUR_API_KEY",
    # API_URL="https://gpt4-o-250120-temp.openai.azure.com/",
    # API_KEY="YOUR_API_KEY",
    # API_URL="https://gpt4o-mini-0226.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview",
    # API_KEY="YOUR_API_KEY",
)

register_model_registry(
    name="azure_4o_registry",
    model_name="gpt-4o@azure",
    raw_model_name="gpt-4o",
    port=-1,
    # API_URL="https://gpt4o-mini-0226.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview",
    # API_KEY="YOUR_API_KEY",
    API_URL="https://bairu-m5yhc9jl-westus.cognitiveservices.azure.com/",
    API_KEY="YOUR_API_KEY",
)

register_model_registry(
    name="azure_o1mini_registry",
    model_name="o1-mini@azure",
    raw_model_name="o1-mini",
    port=-1,
    API_URL="https://gpt4o-mini-0226.openai.azure.com/openai/deployments/o1-mini/chat/completions?api-version=2025-01-01-preview",
    API_KEY="YOUR_API_KEY",
)

register_model_registry(
    name="azure_o4mini_registry",
    model_name="o4-mini@azure",
    raw_model_name="o4-mini",
    port=-1,
    # API_URL="https://bairu-m9m3m6ze-eastus2.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview",
    # API_KEY="YOUR_API_KEY",
    # API_URL="https://gpt4-o-250120-temp.openai.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview",
    # API_URL="https://bairu-m9m3m6ze-eastus2.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview",
    # API_KEY="YOUR_API_KEY",
    API_URL="https://bairu-ma6fsd6p-eastus2.cognitiveservices.azure.com/",
    API_KEY="YOUR_API_KEY",
    # API_KEY="YOUR_API_KEY",
    # API_URL="https://bairu-m5yhc9jl-westus.cognitiveservices.azure.com/openai/deployments/o4-mini/chat/completions?api-version=2025-01-01-preview",
    # API_KEY="YOUR_API_KEY",
)


register_model_registry(
    name="my_claude35haiku_registry",
    model_name="claude-3-5-haiku-20241022",
    raw_model_name="claude-3-5-haiku-20241022",
    port=-1,
    API_KEY="YOUR_API_KEY",
)


register_model_registry(
    name="r1gpt@registry",
    model_name="deepseek-reasoner",
    raw_model_name="gpt-4o-mini",
    port=-1,
    API_URL="https://api.deepseek.com",
    API_KEY="YOUR_API_KEY",
)

register_model_registry(
    name="grok-qwq@registry",
    model_name="qwen-qwq-32b@grok",
    raw_model_name="Qwen/QwQ-32B",
    port=-1,
    API_URL="https://api.groq.com/openai/v1",
    API_KEY="YOUR_API_KEY",
)

register_model_registry(
    name="qwen-2.5-1m@registry",
    model_name="Qwen/Qwen2.5-7B-Instruct-1M@localvllm",
    raw_model_name="Qwen/Qwen2.5-7B-Instruct-1M",
    port=-1,
    API_KEY="",
    API_URL="",
)
