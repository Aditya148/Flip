"""
Example demonstrating logging and debugging utilities.

This example shows:
1. Setting up logging
2. Using the debugger
3. Profiling queries
4. Inspecting components
"""

from flip import Flip, FlipConfig
from flip.utils.logger import setup_logging
from flip.utils.debug import FlipDebugger, profile_query, inspect_chunks
from pathlib import Path


def example_logging():
    """Demonstrate logging setup."""
    print("=" * 60)
    print("Example 1: Logging Setup")
    print("=" * 60)
    
    # Setup logging with different levels
    logger = setup_logging(
        level="DEBUG",
        log_file=Path("./flip_logs/flip.log"),
        console=True
    )
    
    logger.info("Flip SDK initialized")
    logger.debug("This is a debug message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    print("\n✅ Logs written to ./flip_logs/flip.log")


def example_debugging():
    """Demonstrate debugging utilities."""
    print("\n" + "=" * 60)
    print("Example 2: Debugging")
    print("=" * 60)
    
    # Create debugger
    debugger = FlipDebugger(enabled=True, output_dir=Path("./flip_debug"))
    
    # Trace events
    debugger.trace("example", "start", {"message": "Starting example"})
    
    # Time operations
    debugger.start_timer("example_operation")
    
    # Simulate some work
    config = FlipConfig(llm_provider="openai")
    flip = Flip(directory="./sample_docs", config=config)
    
    debugger.stop_timer("example_operation")
    
    debugger.trace("example", "end", {"message": "Example complete"})
    
    # Print timings
    debugger.print_timings()
    
    # Dump to files
    debugger.dump_traces()
    debugger.dump_timings()
    
    print("\n✅ Debug output saved to ./flip_debug/")


def example_profiling():
    """Demonstrate query profiling."""
    print("\n" + "=" * 60)
    print("Example 3: Query Profiling")
    print("=" * 60)
    
    flip = Flip(directory="./sample_docs")
    
    # Profile a query
    profile_query(
        flip_instance=flip,
        query="What is artificial intelligence?",
        iterations=5
    )


def example_component_inspection():
    """Demonstrate component inspection."""
    print("\n" + "=" * 60)
    print("Example 4: Component Inspection")
    print("=" * 60)
    
    from flip.document_processing.loader import DocumentLoader
    from flip.document_processing.chunker import TextChunker
    
    # Load and chunk documents
    loader = DocumentLoader()
    documents = loader.load_directory("./sample_docs", recursive=False)
    
    if documents:
        chunker = TextChunker(strategy="semantic", chunk_size=512)
        chunks = chunker.chunk_text(documents[0].content, documents[0].metadata)
        
        # Inspect chunks
        inspect_chunks(chunks, output_file=Path("./flip_debug/chunk_inspection.json"))


def example_with_logging_and_debugging():
    """Demonstrate using logging and debugging together."""
    print("\n" + "=" * 60)
    print("Example 5: Logging + Debugging")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging(
        level="INFO",
        log_file=Path("./flip_logs/combined.log"),
        console=True
    )
    
    # Setup debugging
    debugger = FlipDebugger(enabled=True)
    
    logger.info("Starting Flip SDK with debugging enabled")
    
    debugger.start_timer("full_pipeline")
    
    # Initialize Flip
    logger.info("Initializing Flip...")
    flip = Flip(directory="./sample_docs")
    
    # Query
    logger.info("Running query...")
    debugger.start_timer("query")
    
    response = flip.query("What is AI?")
    
    debugger.stop_timer("query")
    logger.info(f"Query completed. Answer length: {len(response.answer)}")
    
    debugger.stop_timer("full_pipeline")
    
    # Show results
    debugger.print_timings()
    
    logger.info("Example complete")


def main():
    """Run all examples."""
    print("\n🚀 Flip SDK - Logging & Debugging Examples\n")
    
    try:
        example_logging()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_debugging()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_profiling()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_component_inspection()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_with_logging_and_debugging()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Examples complete!")
    print("\nOutput locations:")
    print("  Logs: ./flip_logs/")
    print("  Debug: ./flip_debug/")


if __name__ == "__main__":
    main()
