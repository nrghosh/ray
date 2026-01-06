.. _working-with-llms:

Working with LLMs
=================

The :ref:`ray.data.llm <llm-ref>` module integrates with LLM inference engines (vLLM, SGLang) to enable scalable batch inference on Ray Data datasets.

**Getting started:**

* :ref:`Quickstart <vllm_quickstart>` - Run your first batch inference job
* :ref:`Architecture <processor_architecture>` - Understand the processor pipeline
* :ref:`Scaling <horizontal_scaling>` - Scale your LLM stage to multiple replicas

**Common use cases:**

* :ref:`Text generation <text_generation>` - Chat completions with LLMs
* :ref:`Embeddings <embedding_models>` - Generate text embeddings
* :ref:`Multimodal models <multimodal>` - Process images, video, and audio with VLMs
* :ref:`OpenAI-compatible endpoints <openai_compatible_api_endpoint>` - Query deployed models

**Operations:**

* :ref:`Troubleshooting <troubleshooting>` - GPU memory, model loading issues
* :ref:`Advanced configuration <advanced_configuration>` - Parallelism, per-stage tuning, LoRA

.. _vllm_quickstart:

Quickstart: vLLM batch inference
---------------------------------

Get started with vLLM batch inference in just a few steps. This example shows the minimal setup needed to run batch inference on a dataset.

.. note::
    This quickstart requires a GPU as vLLM is GPU-accelerated.

First, install Ray Data with LLM support:

.. code-block:: bash

    pip install -U "ray[data, llm]>=2.49.1"

Here's a complete minimal example that runs batch inference:

.. literalinclude:: doc_code/working-with-llms/minimal_quickstart.py
    :language: python
    :start-after: __minimal_vllm_quickstart_start__
    :end-before: __minimal_vllm_quickstart_end__

This example:

1. Creates a simple dataset with prompts
2. Configures a vLLM processor with minimal settings
3. Builds a processor that handles preprocessing (converting prompts to OpenAI chat format) and postprocessing (extracting generated text)
4. Runs inference on the dataset
5. Iterates through results

The processor expects input rows with a ``prompt`` field and outputs rows with both ``prompt`` and ``response`` fields. You can consume results using ``iter_rows()``, ``take()``, ``show()``, or save to files with ``write_parquet()``.

For more configuration options and advanced features, see the sections below.

.. _processor_architecture:

Processor architecture
----------------------

Ray Data LLM uses a **multi-stage processor pipeline** to transform your data through LLM inference. Understanding this architecture helps you optimize performance and debug issues.

.. code-block:: text

    Input Dataset
         |
         v
    - Preprocess (Custom Function)
    - PrepareMultimodal (Optional, for VLMs)
    - ChatTemplate (Applies chat template to messages)
    - Tokenize (Converts text to token IDs)
    - LLM Engine (vLLM/SGLang inference on GPU)
    - Detokenize (Converts token IDs back to text)
    - Postprocess (Custom Function)
         |
         v
    Output Dataset

**Stage descriptions:**

- **Preprocess**: Your custom function that transforms input rows into the format expected by downstream stages (typically OpenAI chat format with ``messages``).
- **PrepareMultimodal**: Extracts and prepares multimodal inputs (images, audio, video). Enable with ``prepare_multimodal_stage={"enabled": True}``.
- **ChatTemplate**: Applies the model's chat template to convert messages into a prompt string.
- **Tokenize**: Converts the prompt string into token IDs for the model.
- **LLM Engine**: The GPU-accelerated inference stage running vLLM or SGLang.
- **Detokenize**: Converts output token IDs back to readable text.
- **Postprocess**: Your custom function that extracts and formats the final output.

Each stage runs as a separate Ray actor pool, enabling independent scaling and resource allocation. CPU stages (ChatTemplate, Tokenize, Detokenize) use autoscaling actor pools, while the GPU stage uses a fixed pool.

.. _horizontal_scaling:

Scaling to multiple GPUs
------------------------

Horizontally scale the LLM stage to multiple GPU replicas using the ``concurrency`` parameter:

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __concurrent_config_example_start__
    :end-before: __concurrent_config_example_end__

Each replica runs an independent inference engine. Set ``concurrency`` to match the number of available GPUs or GPU nodes.

.. _text_generation:

Text generation
---------------

Use :class:`vLLMEngineProcessorConfig <ray.data.llm.vLLMEngineProcessorConfig>` for chat completions and text generation tasks.

**Key configuration options:**

- ``model_source``: HuggingFace model ID or path to model weights
- ``concurrency``: Number of vLLM engine replicas (typically 1 per GPU node)
- ``batch_size``: Rows per batch (reduce if hitting memory limits)

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __basic_config_example_start__
    :end-before: __basic_config_example_end__

For gated models requiring authentication, pass your HuggingFace token through ``runtime_env``:

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __hf_token_config_example_start__
    :end-before: __hf_token_config_example_end__

.. _vllm_llm:

Configure vLLM for LLM inference
--------------------------------

Use the :class:`vLLMEngineProcessorConfig <ray.data.llm.vLLMEngineProcessorConfig>` to configure the vLLM engine.

For handling larger models, specify model parallelism:

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __parallel_config_example_start__
    :end-before: __parallel_config_example_end__

The underlying :class:`Processor <ray.data.llm.Processor>` object instantiates replicas of the vLLM engine and automatically
configure parallel workers to handle model parallelism (for tensor parallelism and pipeline parallelism,
if specified).

To optimize model loading, you can configure the `load_format` to `runai_streamer` or `tensorizer`.

.. note::
    In this case, install vLLM with runai dependencies: `pip install -U "vllm[runai]>=0.10.1"`

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __runai_config_example_start__
    :end-before: __runai_config_example_end__

If your model is hosted on AWS S3, you can specify the S3 path in the ``model_source`` argument and ``load_format="runai_streamer"`` in the ``engine_kwargs`` argument.

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __s3_config_example_start__
    :end-before: __s3_config_example_end__

To do multi-LoRA batch inference, you need to set LoRA related parameters in `engine_kwargs`. See :doc:`the vLLM with LoRA example</llm/examples/batch/vllm-with-lora>` for details.

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __lora_config_example_start__
    :end-before: __lora_config_example_end__

.. _stage_configuration:

Configure processing stages
---------------------------

Each stage in the processor pipeline can be individually configured for fine-grained control over resources and behavior. This is useful when you need to:

- Adjust batch sizes per stage for memory optimization
- Scale CPU stages independently from the GPU stage
- Set different runtime environments per stage
- Control CPU and memory allocation per worker

**Stage configuration options:**

You can configure stages using boolean values, dictionaries, or typed config objects:

.. code-block:: python

    from ray.data.llm import vLLMEngineProcessorConfig

    # Simple: enable/disable with boolean
    config = vLLMEngineProcessorConfig(
        model_source="meta-llama/Llama-3.1-8B-Instruct",
        chat_template_stage=True,       # Enable (default)
        tokenize_stage=True,            # Enable (default)
        detokenize_stage=True,          # Enable (default)
        prepare_multimodal_stage=False, # Disable (default)
    )

    # Advanced: per-stage control with dict
    config = vLLMEngineProcessorConfig(
        model_source="meta-llama/Llama-3.1-8B-Instruct",
        chat_template_stage={
            "enabled": True,
            "batch_size": 256,          # Override batch size for this stage
            "concurrency": 4,           # Scale this stage independently
        },
        tokenize_stage={
            "enabled": True,
            "batch_size": 512,
            "num_cpus": 0.5,            # CPU allocation per worker
        },
        detokenize_stage={
            "enabled": True,
            "concurrency": (2, 8),      # Autoscaling pool from 2-8 workers
        },
    )

**Available stage config fields:**

All stages support these common fields:

- ``enabled`` (bool): Enable or disable the stage
- ``batch_size`` (int): Number of rows per batch for this stage
- ``concurrency`` (int or tuple): Actor pool size; tuple ``(min, max)`` enables autoscaling
- ``runtime_env`` (dict): Runtime environment for this stage's workers
- ``num_cpus`` (float): CPUs to reserve per worker
- ``memory`` (float): Heap memory in bytes per worker

Stage-specific fields:

- ``chat_template_stage``: Also accepts ``chat_template`` (str) and ``chat_template_kwargs`` (dict)
- ``tokenize_stage``, ``detokenize_stage``: Also accept ``model_source`` (str) to use a different tokenizer

When a stage config field isn't specified, it inherits from the processor-level defaults (``batch_size``, ``concurrency``, ``runtime_env``).

.. _multimodal:

Multimodal batch inference
--------------------------------------------------------

Ray Data LLM also supports running batch inference with vision language
and omni-modal models on multimodal data. To enable multimodal batch inference,
apply the following 2 adjustments on top of the previous example:

- Set ``prepare_multimodal_stage={"enabled": True}`` in the ``vLLMEngineProcessorConfig``
- Prepare multimodal data inside the preprocessor.

Prior to running the examples below, install the required dependencies:

.. code-block:: bash

    # Install required dependencies for downloading datasets from Hugging Face
    pip install datasets>=4.0.0

Image batch inference with vision language model (VLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load a vision dataset:

.. literalinclude:: doc_code/working-with-llms/vlm_image_example.py
    :language: python
    :start-after: def load_vision_dataset():
    :end-before: def create_vlm_config():
    :dedent: 0

Next, configure the VLM processor with the essential settings:

.. literalinclude:: doc_code/working-with-llms/vlm_image_example.py
    :language: python
    :start-after: __vlm_config_example_start__
    :end-before: __vlm_config_example_end__

Define preprocessing and postprocessing functions to convert dataset rows into
the format expected by the VLM and extract model responses. Within the preprocessor,
structure image data as part of an OpenAI-compatible message. Both image URL and
``PIL.Image.Image`` object are supported.

.. literalinclude:: doc_code/working-with-llms/vlm_image_example.py
    :language: python
    :start-after: __image_message_format_example_start__
    :end-before: __image_message_format_example_end__

.. literalinclude:: doc_code/working-with-llms/vlm_image_example.py
    :language: python
    :start-after: __vlm_preprocess_example_start__
    :end-before: __vlm_preprocess_example_end__

Finally, run the VLM inference:

.. literalinclude:: doc_code/working-with-llms/vlm_image_example.py
    :language: python
    :start-after: def run_vlm_example():
    :end-before: # __vlm_run_example_end__
    :dedent: 0

Video batch inference with vision language model (VLM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load a video dataset:

.. literalinclude:: doc_code/working-with-llms/vlm_video_example.py
    :language: python
    :start-after: def load_video_dataset():
    :end-before: def create_vlm_video_config():
    :dedent: 0

Next, configure the VLM processor with the essential settings:

.. literalinclude:: doc_code/working-with-llms/vlm_video_example.py
    :language: python
    :start-after: __vlm_video_config_example_start__
    :end-before: __vlm_video_config_example_end__

Define preprocessing and postprocessing functions to convert dataset rows into
the format expected by the VLM and extract model responses. Within the preprocessor,
structure video data as part of an OpenAI-compatible message.

.. literalinclude:: doc_code/working-with-llms/vlm_video_example.py
    :language: python
    :start-after: __vlm_video_preprocess_example_start__
    :end-before: __vlm_video_preprocess_example_end__

Finally, run the VLM inference:

.. literalinclude:: doc_code/working-with-llms/vlm_video_example.py
    :language: python
    :start-after: def run_vlm_video_example():
    :end-before: # __vlm_video_run_example_end__
    :dedent: 0

Audio batch inference with omni-modal model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, load an audio dataset:

.. literalinclude:: doc_code/working-with-llms/omni_audio_example.py
    :language: python
    :start-after: def load_audio_dataset():
    :end-before: def create_omni_audio_config():
    :dedent: 0

Next, configure the omni-modal processor with the essential settings:

.. literalinclude:: doc_code/working-with-llms/omni_audio_example.py
    :language: python
    :start-after: __omni_audio_config_example_start__
    :end-before: __omni_audio_config_example_end__

Define preprocessing and postprocessing functions to convert dataset rows into
the format expected by the omni-modal model and extract model responses. Within the preprocessor,
structure audio data as part of an OpenAI-compatible message. Both audio URL and audio
binary data are supported.

.. literalinclude:: doc_code/working-with-llms/omni_audio_example.py
    :language: python
    :start-after: __audio_message_format_example_start__
    :end-before: __audio_message_format_example_end__

.. literalinclude:: doc_code/working-with-llms/omni_audio_example.py
    :language: python
    :start-after: __omni_audio_preprocess_example_start__
    :end-before: __omni_audio_preprocess_example_end__

Finally, run the omni-modal inference:

.. literalinclude:: doc_code/working-with-llms/omni_audio_example.py
    :language: python
    :start-after: def run_omni_audio_example():
    :end-before: # __omni_audio_run_example_end__
    :dedent: 0

.. _embedding_models:

Embeddings
----------

For embedding models, set ``task_type="embed"`` and disable chat templating:

.. literalinclude:: doc_code/working-with-llms/embedding_example.py
    :language: python
    :start-after: __embedding_example_start__
    :end-before: __embedding_example_end__

Key differences from text generation:

- Use ``prompt`` input instead of ``messages``
- Access results through ``row["embeddings"]``

.. _classification_models:

Batch inference with classification models
------------------------------------------

Ray Data LLM supports batch inference with sequence classification models, such as content classifiers and sentiment analyzers:

.. literalinclude:: doc_code/working-with-llms/classification_example.py
    :language: python
    :start-after: __classification_example_start__
    :end-before: __classification_example_end__

.. testoutput::
    :options: +MOCK

    {'text': 'lol that was so funny haha', 'edu_score': -0.05}
    {'text': 'Photosynthesis converts light energy...', 'edu_score': 1.73}
    {'text': "Newton's laws describe...", 'edu_score': 2.52}

Key differences for classification models:

- Set ``task_type="classify"`` (or ``task_type="score"`` for scoring models)
- Set ``apply_chat_template=False`` and ``detokenize=False``
- Use direct ``prompt`` input instead of ``messages``
- Access classification logits through ``row["embeddings"]``

For a complete classification configuration example, see:

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __classification_config_example_start__
    :end-before: __classification_config_example_end__

.. _openai_compatible_api_endpoint:

OpenAI-compatible endpoints
---------------------------

Query deployed models with an OpenAI-compatible API:

.. literalinclude:: doc_code/working-with-llms/openai_api_example.py
    :language: python
    :start-after: __openai_example_start__
    :end-before: __openai_example_end__

.. _troubleshooting:

Troubleshooting
---------------

GPU memory and CUDA OOM
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter CUDA out of memory errors, try these strategies:

- **Reduce batch size**: Start with 8-16 and increase gradually
- **Lower ``max_num_batched_tokens``**: Reduce from 4096 to 2048 or 1024
- **Decrease ``max_model_len``**: Use shorter context lengths
- **Set ``gpu_memory_utilization``**: Use 0.75-0.85 instead of default 0.90

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __gpu_memory_config_example_start__
    :end-before: __gpu_memory_config_example_end__

Model loading at scale
~~~~~~~~~~~~~~~~~~~~~~

.. _model_cache:

For large clusters, HuggingFace downloads may be rate-limited. Cache models to S3 or GCS:

.. code-block:: bash

    python -m ray.llm.utils.upload_model \
        --model-source facebook/opt-350m \
        --bucket-uri gs://my-bucket/path/to/model

Then reference the remote path in your config:

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __s3_config_example_start__
    :end-before: __s3_config_example_end__

.. _advanced_configuration:

Advanced configuration
----------------------

Model parallelism
~~~~~~~~~~~~~~~~~

For large models that don't fit on a single GPU, use tensor and pipeline parallelism:

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __parallel_config_example_start__
    :end-before: __parallel_config_example_end__

Cross-node parallelism
~~~~~~~~~~~~~~~~~~~~~~

Ray Data LLM supports cross-node parallelism, including tensor parallelism and pipeline parallelism. Configure the parallelism level through ``engine_kwargs``. The ``distributed_executor_backend`` defaults to ``"ray"`` for cross-node support.

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __cross_node_parallelism_config_example_start__
    :end-before: __cross_node_parallelism_config_example_end__

You can customize the placement group strategy to control how Ray places vLLM engine workers across nodes. While you can specify the degree of tensor and pipeline parallelism, the specific assignment of model ranks to GPUs is managed by the vLLM engine.

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __custom_placement_group_strategy_config_example_start__
    :end-before: __custom_placement_group_strategy_config_example_end__

Per-stage configuration
~~~~~~~~~~~~~~~~~~~~~~~

Configure individual pipeline stages for fine-grained resource control:

.. code-block:: python

    config = vLLMEngineProcessorConfig(
        model_source="meta-llama/Llama-3.1-8B-Instruct",
        chat_template_stage={
            "enabled": True,
            "batch_size": 256,
            "concurrency": 4,
        },
        tokenize_stage={
            "enabled": True,
            "batch_size": 512,
            "num_cpus": 0.5,
        },
        detokenize_stage={
            "enabled": True,
            "concurrency": (2, 8),  # Autoscaling pool
        },
    )

Available fields for all stages: ``enabled``, ``batch_size``, ``concurrency``, ``runtime_env``, ``num_cpus``, ``memory``.

LoRA adapters
~~~~~~~~~~~~~

For multi-LoRA batch inference:

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __lora_config_example_start__
    :end-before: __lora_config_example_end__

See :doc:`the vLLM with LoRA example</llm/examples/batch/vllm-with-lora>` for details.

Accelerated model loading with RunAI streamer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use `RunAI Model Streamer <https://github.com/run-ai/runai-model-streamer>`_ for faster model loading from cloud storage:

.. note::
    Install vLLM with runai dependencies: ``pip install -U "vllm[runai]>=0.10.1"``

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __runai_config_example_start__
    :end-before: __runai_config_example_end__

Serve deployments
~~~~~~~~~~~~~~~~~

Share a vLLM engine across multiple processors using :ref:`Ray Serve <serving-llms>`:

.. literalinclude:: doc_code/working-with-llms/basic_llm_example.py
    :language: python
    :start-after: __shared_vllm_engine_config_example_start__
    :end-before: __shared_vllm_engine_config_example_end__

----

**Usage data collection**: Ray collects anonymous usage data to improve Ray Data LLM. To opt out, see :ref:`Ray usage stats <ref-usage-stats>`.
