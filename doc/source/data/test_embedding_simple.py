#!/usr/bin/env python3
"""
Simple embedding test that closely follows the working test pattern from test_vllm_engine_proc.py.
This version uses the exact same pattern as the confirmed working test.
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
        # Configuration pattern from working test
        processor_config = vLLMEngineProcessorConfig(
            model_source="sentence-transformers/all-MiniLM-L6-v2",  # Small, accessible model
            task_type="embed",  # Critical: set to embed for embeddings
            engine_kwargs=dict(
                enable_prefix_caching=False,
                enable_chunked_prefill=False,
                max_model_len=256,  # Reduced to avoid model length warnings
                enforce_eager=True,  # Skip CUDA graph capturing
            ),
            batch_size=32,  # Increased to avoid batch size warnings
            concurrency=1,
            apply_chat_template=False,  # Simpler - no chat template
            detokenize=False,  # Embeddings don't need detokenization
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
    print("Simple Embedding Test (Based on Working Test Pattern)")
    print("=" * 55)
    
    success = test_simple_embedding()
    
    if success:
        print("\nSimple embedding test passed!")
    else:
        print("\nSimple embedding test failed!")
        exit(1)
