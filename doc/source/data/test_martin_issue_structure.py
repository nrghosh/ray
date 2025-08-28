#!/usr/bin/env python3
"""
Test that reproduces Martin's exact structure from GitHub issue #55384.

This replicates the workaround approach where task type is explicitly passed
in engine_kwargs due to propagation issues.
"""

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor
from ray.llm._internal.batch.processor.vllm_engine_proc import vLLMTaskType


def test_martin_issue_structure():
    """
    Test using Martin's exact structure from the GitHub issue.
    """
    print("Testing Martin's issue structure with explicit task workaround...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    try:
        # Martin's structure from the issue
        task_type = vLLMTaskType.EMBED
        
        # Explicit workaround - pass task type in engine_kwargs
        # as mentioned in the issue comment
        vllm_engine_args = {}
        vllm_engine_args["task"] = task_type
        vllm_engine_args["max_model_len"] = 256  # Add some basic engine args
        vllm_engine_args["enforce_eager"] = True
        
        model_path = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Martin's config structure
        llm_processor = build_llm_processor(
            vLLMEngineProcessorConfig(
                model_source=model_path,
                engine_kwargs=vllm_engine_args,
                apply_chat_template=False,
                task_type=task_type,  # Also set in config
                concurrency=1,
                batch_size=32,  # Reduced from 256 to avoid warnings
                detokenize=False,
            ),
            preprocess=lambda row: dict(prompt=row["text"]),
            postprocess=lambda row: {
                "text": row["prompt"], 
                "embedding": row["embeddings"]
            },
        )

        # Create test data (instead of reading parquet)
        test_texts = [
            "Hello world",
            "This is a test sentence", 
            "Embedding models convert text to vectors",
        ]
        inputs = ray.data.from_items([{"text": text} for text in test_texts])
        
        # Process (instead of writing parquet)
        print(f"Processing {len(test_texts)} texts...")
        outputs = llm_processor(inputs)
        results = outputs.take_all()
        
        # Validate results
        assert len(results) == len(test_texts), f"Expected {len(test_texts)} results, got {len(results)}"
        assert all("text" in result for result in results), "Missing text in results"
        assert all("embedding" in result for result in results), "Missing embedding in results"
        assert all(result["embedding"] is not None for result in results), "Found null embeddings"

        print(f"Successfully processed {len(results)} texts!")
        
        # Show results
        for i, result in enumerate(results):
            embedding = result["embedding"]
            print(f"Text {i+1}: {result['text']}")
            if hasattr(embedding, '__len__'):
                print(f"  Embedding type: {type(embedding)}")
                if hasattr(embedding, 'shape'):
                    print(f"  Embedding shape: {embedding.shape}")
                elif isinstance(embedding, list):
                    print(f"  Embedding length: {len(embedding)}")
                print(f"  Sample values: {str(embedding)[:100]}...")
            else:
                print(f"  Embedding: {embedding}")
            print()

        print("Martin's issue structure test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error in Martin's structure test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


def test_with_original_problematic_model():
    """
    Test with the original model that was causing issues (Meta-Llama-3-8B-Instruct).
    This should help us see if the task propagation issue is model-specific.
    """
    print("\nTesting with original problematic model (Meta-Llama-3-8B-Instruct)...")
    
    if not ray.is_initialized():
        ray.init()
    
    try:
        task_type = vLLMTaskType.EMBED
        
        # Explicit task in engine_kwargs
        vllm_engine_args = {}
        vllm_engine_args["task"] = task_type
        vllm_engine_args["max_model_len"] = 256
        vllm_engine_args["enforce_eager"] = True
        
        # Original problematic model
        model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        llm_processor = build_llm_processor(
            vLLMEngineProcessorConfig(
                model_source=model_path,
                engine_kwargs=vllm_engine_args,
                apply_chat_template=False,
                task_type=task_type,
                concurrency=1,
                batch_size=32,
                detokenize=False,
            ),
            preprocess=lambda row: dict(prompt=row["text"]),
            postprocess=lambda row: {
                "text": row["prompt"], 
                "embedding": row["embeddings"]
            },
        )

        # Small test
        inputs = ray.data.from_items([{"text": "Hello world"}])
        outputs = llm_processor(inputs)
        results = outputs.take_all()
        
        print("Original model test completed successfully!")
        print(f"Result: {results[0]}")
        return True
        
    except Exception as e:
        print(f"Expected error with original model: {e}")
        # This might fail, which helps us confirm the issue
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    print("Testing Martin's GitHub Issue #55384 Structure")
    print("=" * 50)
    
    # Test 1: Working model with Martin's structure
    success1 = test_martin_issue_structure()
    
    # Test 2: Original problematic model
    success2 = test_with_original_problematic_model()
    
    print(f"\nResults:")
    print(f"Martin's structure (working model): {'PASS' if success1 else 'FAIL'}")
    print(f"Original model test: {'PASS' if success2 else 'FAIL (expected)'}")
    
    if success1:
        print("\nWorkaround approach with explicit task in engine_kwargs works!")
    else:
        print("\nEven the workaround approach has issues!")
