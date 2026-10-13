# ai-image-generator

CUDA-enabled Flask API for text-to-image and multi-reference image generation
with Hugging Face Diffusers. The default model is
`black-forest-labs/FLUX.2-klein-9B`.

Open `http://localhost:9000/` for the browser portal, or use the API directly.
The portal includes persistent generation history with preview, download,
reference reuse, and deletion controls.

Available models, their `text` or `text_image` capability, and the default are defined in `config.yml`. Set `CONFIG_FILE`
to load a different YAML file. `PRETRAINED_MODEL` may override the default when
its value is present in the configured `models` list.

Optional generation styles are configured as `styles` entries with `name` and
`prompt`. When a style is selected, its configured prompt is appended to the
user prompt immediately before generation.

Optional output sizes are configured in the `aspect_ratios` mapping. Each key
is an API and portal option whose value is `[width, height]` in pixels.

Prompt enhancement is enabled by adding a `prompt_enhancement_llm` mapping with
an OpenAI-compatible base `url` (including `/v1`) and `model`. The portal then
shows an **Enhance by LLM** button below the prompt. Requests are sent by the
server to the configured vision-capable `/chat/completions` endpoint. Selected
reference images are included in the enhancement request. The values can be
overridden with `PROMPT_ENHANCEMENT_LLM_URL`,
`PROMPT_ENHANCEMENT_LLM_MODEL`, and `PROMPT_ENHANCEMENT_LLM_API_KEY`.

## Build and run

Create your local configuration before building:

```sh
cp config.example.yml config.yml
```

Edit `config.yml` for your environment, including the prompt enhancement server
URL and model, or remove `prompt_enhancement_llm` to disable that feature.
`config.yml` is ignored by Git; keep private addresses and credentials out of
`config.example.yml`. Supply API keys through environment variables.
The Docker build copies `config.yml` into the image, so use a configuration
without secrets when building an image you intend to share.

The configured models are gated. Accept each model's Hugging Face license that
you intend to use and provide a token that has access to it. The NF4 model also
requires `bitsandbytes`; it is installed by the Docker image.

```sh
docker build -t ai-image-generator .
docker run --gpus all --rm -p 9000:9000 \
  -e HF_TOKEN="$HF_TOKEN" \
  ai-image-generator
```

Change the listening port with `PORT`, for example `-e PORT=8080 -p 8080:8080`.
The Gunicorn worker count is controlled by `WORKERS` and defaults to 1. Each
additional worker loads a separate model copy and requires additional GPU memory.
The model is downloaded on the first generation request. It remains in GPU
memory for `MODEL_IDLE_TIMEOUT` seconds after generation, then is unloaded and
the CUDA cache is released. Its `config.yml` default is 300 seconds (5 minutes);
set it to `0` to keep the model loaded indefinitely. Each new generation resets
the idle timer. Download progress and a periodic elapsed-time message are written
to the container output and can be followed with `docker logs -f <container>`.
The application log level is controlled by `LOG_LEVEL` and defaults to `INFO`;
for example, set `-e LOG_LEVEL=DEBUG` for more verbose logging.
Set `HF_HOME` and mount that directory to persist the downloaded model cache, so
reloading does not require another download.

`GPU_MEMORY_FRACTION` limits each worker's PyTorch CUDA allocator to a fraction
of the visible GPU's total VRAM. Its `config.yml` default is `0.5`, and it accepts
values greater than `0` and at most `1`; for example, `0.5` limits a worker to
approximately 50% of VRAM:

```sh
docker run --gpus all --rm -p 9000:9000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -e GPU_MEMORY_FRACTION=0.5 \
  ai-image-generator
```

This is a hard allocation limit, not automatic CPU offloading. Generation fails
with an out-of-memory error if the model and its intermediate tensors exceed the
limit. The limit applies separately to every worker, so keep `WORKERS=1` when
sharing a GPU.

Environment variables with these names override their values from `config.yml`.

Generated images are also stored under `OUTPUT_DIR`, which defaults to
`/var/ai-image-generator`. Mount that directory to preserve the files outside the
container:

```sh
docker run --gpus all --rm -p 9000:9000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ./generated:/var/ai-image-generator \
  ai-image-generator
```

Each successful generation is saved in a request folder such as
`request_20260831T143000_a1b2c3d4/`. The folder contains `prompt.txt`,
`metadata.json`, normalized reference files named `reference1.png`,
`reference2.png`, and generated files named `image1.png`, `image2.png`, etc.

## API

`POST /generate` accepts `multipart/form-data`:

- `prompt`: optional text prompt
- `style`: optional configured style name; omit it or send an empty string for no style
- `aspect_ratio`: optional configured aspect-ratio name; sets the pipeline width and height
- `image_count`: number of images to generate, from 1 to 4; defaults to 2
- `images`: optional, repeatable image or archive upload
- `pretrained_model` (or `model`): optional model ID from `GET /models`

Sending reference images to a model configured with `type: text` returns a 400
error. Models configured with `type: text_image` accept prompts and images.

At least one prompt or valid image is required. ZIP and TAR-family archives are
supported; every recognized image inside an archive is used as a reference. A
successful request returns JSON describing the saved request and containing URLs
for all generated and reference images.

For text-only generation, a JSON body such as `{"prompt": "..."}` is also
accepted. The repeatable upload field may be named either `images` or `images[]`.

```sh
# Text to image
curl -X POST http://localhost:9000/generate \
  -F 'prompt=A cinematic photograph of Hong Kong in the rain'

# Multiple reference images (archives can be supplied the same way)
curl -X POST http://localhost:9000/generate \
  -F 'prompt=Combine these references into a watercolor illustration' \
  -F 'image_count=2' \
  -F 'images=@reference-1.png' \
  -F 'images=@references.zip'
```

`GET /health` returns process health and the currently loaded model ID.

`GET /models` returns the configured model IDs, a `model_types` capability map,
and the default model.

History endpoints used by the portal are also available directly:

- `GET /api/history`: list stored generation requests, newest first
- `GET /api/history/<request_id>`: retrieve one request's details
- `GET /history/<request_id>/<filename>`: retrieve a generated or reference image
- `DELETE /api/history/<request_id>`: permanently delete a request folder

`POST /api/enhance-prompt` accepts either JSON such as
`{"prompt": "a city at night"}` or multipart form data with `prompt` and
repeatable `images` fields. It returns the enhanced prompt as `{"prompt": "..."}`
when prompt enhancement is configured.

Upload and extraction limits can be changed with `MAX_CONTENT_LENGTH_MB`
(default 100), `MAX_ARCHIVE_FILES` (default 100), and
`MAX_ARCHIVE_SIZE_MB` (default 500).
