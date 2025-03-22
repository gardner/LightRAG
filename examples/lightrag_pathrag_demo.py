import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./pathrag_demo"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        use_pathrag=True  # Enable PathRAG functionality
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def demo_pathrag():
    # Initialize RAG instance with PathRAG enabled
    rag = await initialize_rag()

    # Sample text with relationships to demonstrate path-based retrieval
    sample_text = """
    Alice is the CEO of TechCorp. Bob is the CTO of TechCorp and reports directly to Alice.
    Charlie is a senior developer who works under Bob's supervision. 
    David is a project manager who coordinates with both Bob and Charlie.
    Eve is an external consultant who advises Alice on strategic matters.
    Frank is a customer who has purchased TechCorp's flagship product, which was developed by Charlie's team.
    The flagship product uses a machine learning algorithm designed by Grace, who is a research scientist at TechCorp.
    """

    # Insert the sample text
    print("Inserting sample text...")
    await rag.ainsert(sample_text)
    print("Sample text inserted.")

    # Query using different modes
    print("\n--- Standard Global Mode ---")
    result_global = await rag.aquery(
        "What is the relationship between Alice and Charlie?", 
        param=QueryParam(mode="global")
    )
    print(result_global)

    print("\n--- Path-based Mode ---")
    result_path = await rag.aquery(
        "What is the relationship between Alice and Charlie?", 
        param=QueryParam(mode="path", path_threshold=0.3, path_decay_rate=0.8)
    )
    print(result_path)

    print("\n--- Hybrid Mode with PathRAG ---")
    # With use_pathrag=True, hybrid mode will use path-based retrieval
    result_hybrid = await rag.aquery(
        "How is Frank connected to Grace?", 
        param=QueryParam(mode="hybrid")
    )
    print(result_hybrid)
    
    print("\n--- Complex Relationship Query ---")
    result_complex = await rag.aquery(
        "Describe the full chain of command from Alice to Charlie.", 
        param=QueryParam(mode="path", path_threshold=0.2, path_decay_rate=0.9)
    )
    print(result_complex)
    
    print("\n--- Comparing Retrieval Modes (Context Only) ---")
    # Compare global vs path retrieval to see the difference in contexts
    print("Global mode context:")
    global_info = await rag.aquery(
        "How is Frank connected to Grace?",
        param=QueryParam(mode="global", only_need_context=True)
    )
    print(global_info)
    
    print("\nPath mode context:")
    path_info = await rag.aquery(
        "How is Frank connected to Grace?",
        param=QueryParam(mode="path", only_need_context=True)
    )
    print(path_info)


if __name__ == "__main__":
    asyncio.run(demo_pathrag())