#!/usr/bin/env python3
"""
Replicate martinbomio embedding error / issue

this is currently the config I am using:

        vLLMEngineProcessorConfig(
            model_source=model_path,
            engine_kwargs={},
            apply_chat_template=False,
            task_type="embed",
            concurrency=1,
            batch_size=256,
            detokenize=False
        ),

this is the error I get:

(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634] Traceback (most recent call last):
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]   File "/home/ray/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 625, in run_engine_core
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]     engine_core.run_busy_loop()
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]   File "/home/ray/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 650, in run_busy_loop
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]     self._process_input_queue()
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]   File "/home/ray/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 663, in _process_input_queue
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]     self._handle_client_request(*req)
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]   File "/home/ray/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 689, in _handle_client_request
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]     self.add_request(request)
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]   File "/home/ray/.venv/lib/python3.11/site-packages/vllm/v1/engine/core.py", line 203, in add_request
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634]     raise ValueError(f"Unsupported task: {pooling_params.task!r} "
(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=589, ip=240.248.0.198) ERROR 08-28 03:22:49 [core.py:634] ValueError: Unsupported task: None Supported tasks: ('embed', 'encode')


I am using vllm==0.10.0

I am also using https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct as the model
"""

import ray
from ray.llm._internal.batch.processor import ProcessorBuilder
from ray.llm._internal.batch.processor.vllm_engine_proc import vLLMEngineProcessorConfig


def test_simple_embedding():
    """
    Simple embedding test based on the working test_embedding_model pattern.
    Uses the same configuration pattern as the confirmed working test.
    """
    print("Testing simple embedding functionality...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        # model_path = "sentence-transformers/all-MiniLM-L6-v2"
        # model_path = "meta-llama/Llama-3.1-8B-Instruct"
        # Configuration pattern from working test
        processor_config = vLLMEngineProcessorConfig(
            model_source=model_path,
            engine_kwargs={"max_model_len":128},
            apply_chat_template=False,
            task_type="embed",
            concurrency=1,
            batch_size=256,
            detokenize=False
        )

        # Build processor using the same pattern as working test
        processor = ProcessorBuilder.build(
            processor_config,
            preprocess=lambda row: dict(
                prompt=row["text"],  # Direct prompt input
            ),
            postprocess=lambda row: {
                "original_text": row["prompt"],
                "embedding": row["embeddings"],  # Extract embeddings
            },
        )

        # Create test dataset (same pattern as working test)
        test_data = [
            "Hello world",
            "This is a test sentence",
            "Embedding models convert text to vectors",
            "Ray Data enables scalable ML processing",
        ]
        ds = ray.data.from_items([{"text": text} for text in test_data])
        
        # Process through embedding pipeline
        print(f"Processing {len(test_data)} texts through embedding model...")
        ds = processor(ds)
        ds = ds.materialize()
        results = ds.take_all()

        # Validation (same pattern as working test)
        assert len(results) == len(test_data), f"Expected {len(test_data)} results, got {len(results)}"
        assert all("original_text" in result for result in results), "Missing original_text in results"
        assert all("embedding" in result for result in results), "Missing embedding in results"
        assert all(result["embedding"] is not None for result in results), "Found null embeddings"

        print(f"Successfully processed {len(results)} texts!")
        
        # Show results
        for i, result in enumerate(results):
            embedding = result["embedding"]
            print(f"Text {i+1}: {result['original_text']}")
            if isinstance(embedding, list):
                print(f"  Embedding dimension: {len(embedding)}")
                print(f"  Sample values: {embedding[:3] if len(embedding) >= 3 else embedding}")
            else:
                print(f"  Embedding type: {type(embedding)}")
            print()

        print("Simple embedding test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in simple embedding test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    print("Simple Embedding Test (Based on Martin's Test Pattern / Engine Config)")
    print("=" * 55)
    
    success = test_simple_embedding()
    
    if success:
        print("\nSimple embedding test passed!")
    else:
        print("\nSimple embedding test failed!")
        exit(1)
