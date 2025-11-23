"""
Quick test to verify Flip SDK installation and basic functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from flip import Flip, FlipConfig
        print("✅ Core imports successful")
    except Exception as e:
        print(f"❌ Core imports failed: {e}")
        return False
    
    try:
        from flip.generation.factory import LLMFactory
        from flip.embedding.factory import EmbedderFactory
        from flip.vector_store.factory import VectorStoreFactory
        print("✅ Factory imports successful")
    except Exception as e:
        print(f"❌ Factory imports failed: {e}")
        return False
    
    try:
        from flip.document_processing.loader import DocumentLoader
        from flip.document_processing.chunker import TextChunker
        from flip.document_processing.preprocessor import TextPreprocessor
        print("✅ Document processing imports successful")
    except Exception as e:
        print(f"❌ Document processing imports failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    
    try:
        from flip import FlipConfig
        
        config = FlipConfig()
        print(f"✅ Default config created")
        print(f"   LLM Provider: {config.llm_provider}")
        print(f"   LLM Model: {config.get_llm_default_model()}")
        print(f"   Embedding Provider: {config.embedding_provider}")
        print(f"   Embedding Model: {config.get_embedding_default_model()}")
        
        # Test custom config
        custom_config = FlipConfig(
            llm_provider="anthropic",
            embedding_provider="sentence-transformers",
            chunking_strategy="semantic"
        )
        print(f"✅ Custom config created")
        print(f"   LLM Provider: {custom_config.llm_provider}")
        print(f"   Chunking Strategy: {custom_config.chunking_strategy}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_factories():
    """Test factory pattern."""
    print("\nTesting factories...")
    
    try:
        from flip.generation.factory import LLMFactory
        from flip.embedding.factory import EmbedderFactory
        from flip.vector_store.factory import VectorStoreFactory
        
        # Test LLM factory
        llm_providers = LLMFactory.get_supported_providers()
        print(f"✅ LLM Factory - Supported providers: {', '.join(llm_providers)}")
        
        # Test Embedder factory
        embedding_providers = EmbedderFactory.get_supported_providers()
        print(f"✅ Embedder Factory - Supported providers: {', '.join(embedding_providers)}")
        
        # Test Vector Store factory
        vector_providers = VectorStoreFactory.get_supported_providers()
        print(f"✅ Vector Store Factory - Supported providers: {', '.join(vector_providers)}")
        
        return True
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False


def test_document_processing():
    """Test document processing components."""
    print("\nTesting document processing...")
    
    try:
        from flip.document_processing.chunker import TextChunker
        from flip.document_processing.preprocessor import TextPreprocessor
        
        # Test preprocessor
        preprocessor = TextPreprocessor()
        text = "  This   is  a   test   text.  "
        cleaned = preprocessor.clean_text(text)
        print(f"✅ Text preprocessing works")
        print(f"   Original: '{text}'")
        print(f"   Cleaned: '{cleaned}'")
        
        # Test chunker
        chunker = TextChunker(strategy="token", chunk_size=50)
        test_text = "This is a test document. " * 20
        chunks = chunker.chunk_text(test_text, {"source": "test"})
        print(f"✅ Text chunking works")
        print(f"   Input length: {len(test_text)} chars")
        print(f"   Chunks created: {len(chunks)}")
        
        return True
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_supported_extensions():
    """Test supported file extensions."""
    print("\nTesting supported file extensions...")
    
    try:
        from flip.document_processing.loader import DocumentLoader
        
        extensions = DocumentLoader.get_supported_extensions()
        print(f"✅ Supported extensions ({len(extensions)}):")
        print(f"   {', '.join(extensions)}")
        
        return True
    except Exception as e:
        print(f"❌ Extensions test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Flip SDK - Quick Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_factories,
        test_document_processing,
        test_supported_extensions,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
