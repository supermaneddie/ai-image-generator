import os
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yml")


def _float_setting(config: dict, name: str) -> float:
    value = os.getenv(name, config.get(name))
    if value is None:
        raise ValueError(f"Configuration '{name}' is required")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number") from error


def _load_config() -> dict:
    config_path = Path(os.getenv("CONFIG_FILE", DEFAULT_CONFIG_PATH))
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"Could not load configuration from {config_path}: {error}") from error

    models = config.get("models")
    default_model = config.get("default_model")
    if not isinstance(models, list) or not models:
        raise ValueError("Configuration 'models' must be a non-empty list")

    model_types = {}
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("Each configured model must define 'id' and 'type'")
        model_id = model.get("id")
        model_type = model.get("type")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Each configured model must have a non-empty 'id'")
        if model_type not in {"text", "text_image"}:
            raise ValueError("Each configured model 'type' must be 'text' or 'text_image'")
        model_types[model_id.strip()] = model_type

    model_ids = [model["id"].strip() for model in models]
    if len(model_ids) != len(model_types):
        raise ValueError("Configuration 'models' must not contain duplicates")
    if default_model not in model_types:
        raise ValueError("Configuration 'default_model' must be included in 'models'")

    configured_default = os.getenv("PRETRAINED_MODEL", default_model).strip()
    if configured_default not in model_types:
        raise ValueError("PRETRAINED_MODEL must name a model defined in the configuration")

    styles = config.get("styles", [])
    if not isinstance(styles, list):
        raise ValueError("Configuration 'styles' must be a list")

    style_prompts = {}
    for style in styles:
        if not isinstance(style, dict):
            raise ValueError("Each configured style must define 'name' and 'prompt'")
        name = style.get("name")
        prompt = style.get("prompt")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each configured style must have a non-empty 'name'")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each configured style must have a non-empty 'prompt'")
        normalized_name = name.strip()
        if normalized_name in style_prompts:
            raise ValueError("Configuration 'styles' must not contain duplicate names")
        style_prompts[normalized_name] = prompt.strip()

    aspect_ratios = config.get("aspect_ratios", {})
    if not isinstance(aspect_ratios, dict):
        raise ValueError("Configuration 'aspect_ratios' must be a mapping")

    normalized_aspect_ratios = {}
    for name, dimensions in aspect_ratios.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each configured aspect ratio must have a non-empty name")
        if (
            not isinstance(dimensions, (list, tuple))
            or len(dimensions) != 2
            or not all(isinstance(value, int) and value > 0 for value in dimensions)
        ):
            raise ValueError(
                "Each configured aspect ratio must contain positive integer width and height"
            )
        normalized_aspect_ratios[name.strip()] = tuple(dimensions)

    model_idle_timeout = _float_setting(config, "MODEL_IDLE_TIMEOUT")
    if model_idle_timeout < 0:
        raise ValueError("MODEL_IDLE_TIMEOUT must be greater than or equal to 0")

    gpu_memory_fraction = _float_setting(config, "GPU_MEMORY_FRACTION")
    if not 0 < gpu_memory_fraction <= 1:
        raise ValueError("GPU_MEMORY_FRACTION must be greater than 0 and at most 1")

    prompt_enhancement_llm = config.get("prompt_enhancement_llm")
    if prompt_enhancement_llm is not None:
        if not isinstance(prompt_enhancement_llm, dict):
            raise ValueError("Configuration 'prompt_enhancement_llm' must be a mapping")
        llm_url = os.getenv(
            "PROMPT_ENHANCEMENT_LLM_URL", prompt_enhancement_llm.get("url", "")
        )
        llm_model = os.getenv(
            "PROMPT_ENHANCEMENT_LLM_MODEL", prompt_enhancement_llm.get("model", "")
        )
        if not isinstance(llm_url, str) or not llm_url.strip():
            raise ValueError(
                "Configuration 'prompt_enhancement_llm' must define non-empty 'url' and 'model'"
            )
        if not isinstance(llm_model, str) or not llm_model.strip():
            raise ValueError(
                "Configuration 'prompt_enhancement_llm' must define non-empty 'url' and 'model'"
            )
        prompt_enhancement_llm = {
            "url": llm_url.strip().rstrip("/"),
            "model": llm_model.strip(),
            "api_key": os.getenv(
                "PROMPT_ENHANCEMENT_LLM_API_KEY",
                prompt_enhancement_llm.get("api_key") or "",
            ).strip(),
        }

    return {
        "models": tuple(model_ids),
        "model_types": model_types,
        "styles": tuple(style_prompts),
        "style_prompts": style_prompts,
        "aspect_ratios": normalized_aspect_ratios,
        "default_model": configured_default,
        "model_idle_timeout": model_idle_timeout,
        "gpu_memory_fraction": gpu_memory_fraction,
        "prompt_enhancement_llm": prompt_enhancement_llm,
    }


_CONFIG = _load_config()
AVAILABLE_MODELS = _CONFIG["models"]
MODEL_TYPES = _CONFIG["model_types"]
AVAILABLE_STYLES = _CONFIG["styles"]
STYLE_PROMPTS = _CONFIG["style_prompts"]
ASPECT_RATIOS = _CONFIG["aspect_ratios"]
DEFAULT_MODEL = _CONFIG["default_model"]
MODEL_IDLE_TIMEOUT = _CONFIG["model_idle_timeout"]
GPU_MEMORY_FRACTION = _CONFIG["gpu_memory_fraction"]
PROMPT_ENHANCEMENT_LLM = _CONFIG["prompt_enhancement_llm"]
