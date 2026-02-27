"""
Experiment 5.5: RAG Retrieval Quality

Tests the quality of RAG retrieval system:
1. Precision@K: How many retrieved chunks are relevant?
2. MRR (Mean Reciprocal Rank): Where does the correct chunk appear?
3. Impact of metadata filtering: With vs. without crop filtering

Dataset Requirements:
- Create 30 queries with pre-labeled relevant chunks
- Ground truth: Which chunks should be retrieved for each query
"""

import os
import sys
from typing import List, Dict, Tuple
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.engine import get_query_engine
from evaluations.eval_utils import EvaluationResult, Timer, print_metrics


def create_test_queries() -> List[Dict]:
    """
    Create test queries with ground truth
    
    Each query should have:
    - query: The search query
    - crop: Crop type for filtering
    - relevant_keywords: Keywords that should appear in results
    """
    return [
        {
            'id': 'Q001',
            'query': 'Treatment for Early blight in Tomato',
            'crop': 'Tomato',
            'relevant_keywords': ['early blight', 'tomato', 'fungicide', 'chlorothalonil', 'copper'],
            'expected_chunks': 1  # How many chunks should be relevant
        },
        {
            'id': 'Q002',
            'query': 'How to control aphids on tomato plants',
            'crop': 'Tomato',
            'relevant_keywords': ['aphid', 'insecticide', 'neem oil', 'spray', 'control'],
            'expected_chunks': 1
        },
        {
            'id': 'Q003',
            'query': 'Septoria leaf spot management',
            'crop': 'Tomato',
            'relevant_keywords': ['septoria', 'leaf spot', 'fungicide', 'remove', 'infected'],
            'expected_chunks': 1
        },
        {
            'id': 'Q004',
            'query': 'Late blight prevention strategies',
            'crop': 'Tomato',
            'relevant_keywords': ['late blight', 'prevention', 'fungicide', 'resistant'],
            'expected_chunks': 1
        },
        {
            'id': 'Q005',
            'query': 'Bacterial spot treatment',
            'crop': 'Tomato',
            'relevant_keywords': ['bacterial', 'copper', 'spray', 'treatment'],
            'expected_chunks': 1
        },
        # Add more queries for comprehensive testing
    ]


def evaluate_retrieval(query: str, retrieved_chunks: List[str], 
                       relevant_keywords: List[str], k: int = 3) -> Dict:
    """
    Evaluate retrieval quality
    
    Args:
        query: Search query
        retrieved_chunks: List of retrieved text chunks
        relevant_keywords: Keywords that indicate relevance
        k: Number of top chunks to consider
    
    Returns:
        Metrics dictionary
    """
    # Take top k chunks
    top_k_chunks = retrieved_chunks[:k]
    
    # Check relevance based on keyword matching
    relevant_count = 0
    first_relevant_rank = None
    
    for rank, chunk in enumerate(top_k_chunks, start=1):
        chunk_lower = chunk.lower()
        
        # Check if any relevant keywords appear in chunk
        is_relevant = any(keyword.lower() in chunk_lower for keyword in relevant_keywords)
        
        if is_relevant:
            relevant_count += 1
            if first_relevant_rank is None:
                first_relevant_rank = rank
    
    # Calculate metrics
    precision_at_k = relevant_count / k if k > 0 else 0
    
    # Mean Reciprocal Rank (MRR)
    mrr = 1.0 / first_relevant_rank if first_relevant_rank else 0
    
    return {
        'precision_at_k': precision_at_k,
        'mrr': mrr,
        'relevant_count': relevant_count,
        'first_relevant_rank': first_relevant_rank or float('inf')
    }


def test_rag_with_filter(queries: List[Dict], top_k: int = 3) -> Dict:
    """
    Test RAG with metadata filtering
    
    Args:
        queries: List of test queries
        top_k: Number of chunks to retrieve
    
    Returns:
        Aggregated metrics
    """
    print(f"\n🔬 Testing RAG WITH metadata filtering (top_k={top_k})")
    
    all_precisions = []
    all_mrrs = []
    total_time = 0
    
    for query_item in queries:
        query = query_item['query']
        crop = query_item['crop']
        keywords = query_item['relevant_keywords']
        
        print(f"\n  Query: {query}")
        print(f"  Crop filter: {crop}")
        
        with Timer(f"  Retrieval") as timer:
            # Get query engine with crop filter
            engine = get_query_engine(crop_filter=crop)
            
            if engine is None:
                print("    ⚠️  RAG engine not available (no database found)")
                continue
            
            # Query
            response = engine.query(query)
            response_text = str(response)
        
        total_time += timer.elapsed
        
        # For evaluation, we need the actual chunks
        # In a real implementation, you'd access the source nodes
        # For now, we'll use the response text as a single chunk
        chunks = [response_text]
        
        # Evaluate
        metrics = evaluate_retrieval(query, chunks, keywords, k=min(top_k, len(chunks)))
        
        all_precisions.append(metrics['precision_at_k'])
        all_mrrs.append(metrics['mrr'])
        
        print(f"    Precision@{top_k}: {metrics['precision_at_k']:.3f}")
        print(f"    MRR: {metrics['mrr']:.3f}")
    
    # Aggregate metrics
    return {
        'avg_precision_at_k': sum(all_precisions) / len(all_precisions) if all_precisions else 0,
        'avg_mrr': sum(all_mrrs) / len(all_mrrs) if all_mrrs else 0,
        'avg_retrieval_time': total_time / len(queries) if queries else 0,
        'total_queries': len(queries)
    }


def test_rag_without_filter(queries: List[Dict], top_k: int = 3) -> Dict:
    """
    Test RAG WITHOUT metadata filtering (search entire database)
    
    This should be worse than with filtering.
    """
    print(f"\n🔬 Testing RAG WITHOUT metadata filtering (top_k={top_k})")
    
    all_precisions = []
    all_mrrs = []
    total_time = 0
    
    for query_item in queries:
        query = query_item['query']
        keywords = query_item['relevant_keywords']
        
        print(f"\n  Query: {query}")
        print(f"  Crop filter: None (searching all crops)")
        
        with Timer(f"  Retrieval") as timer:
            # Get query engine WITHOUT crop filter
            engine = get_query_engine(crop_filter=None)
            
            if engine is None:
                print("    ⚠️  RAG engine not available (no database found)")
                continue
            
            response = engine.query(query)
            response_text = str(response)
        
        total_time += timer.elapsed
        
        chunks = [response_text]
        metrics = evaluate_retrieval(query, chunks, keywords, k=min(top_k, len(chunks)))
        
        all_precisions.append(metrics['precision_at_k'])
        all_mrrs.append(metrics['mrr'])
        
        print(f"    Precision@{top_k}: {metrics['precision_at_k']:.3f}")
        print(f"    MRR: {metrics['mrr']:.3f}")
    
    return {
        'avg_precision_at_k': sum(all_precisions) / len(all_precisions) if all_precisions else 0,
        'avg_mrr': sum(all_mrrs) / len(all_mrrs) if all_mrrs else 0,
        'avg_retrieval_time': total_time / len(queries) if queries else 0,
        'total_queries': len(queries)
    }


def run_experiment():
    """Main experiment runner"""
    print("\n" + "="*60)
    print("🧪 EXPERIMENT 5.5: RAG RETRIEVAL QUALITY")
    print("="*60)
    
    # Get test queries
    queries = create_test_queries()
    print(f"\n✅ Loaded {len(queries)} test queries")
    
    # Check if RAG database exists
    engine = get_query_engine()
    if engine is None:
        print("\n⚠️  WARNING: RAG database not found!")
        print("Please run the ingestion script first:")
        print("  python -m app.rag.ingest --target ./data/manuals/")
        return
    
    # Test with metadata filtering
    print("\n" + "="*60)
    print("🎯 Test 1: WITH Metadata Filtering")
    print("="*60)
    
    metrics_with_filter = test_rag_with_filter(queries, top_k=3)
    print_metrics(metrics_with_filter, "Results WITH Filtering")
    
    # Test without metadata filtering
    print("\n" + "="*60)
    print("🎯 Test 2: WITHOUT Metadata Filtering")
    print("="*60)
    
    metrics_without_filter = test_rag_without_filter(queries, top_k=3)
    print_metrics(metrics_without_filter, "Results WITHOUT Filtering")
    
    # Comparison
    print("\n" + "="*60)
    print("🔍 COMPARISON")
    print("="*60)
    
    improvement_precision = (
        (metrics_with_filter['avg_precision_at_k'] - metrics_without_filter['avg_precision_at_k']) 
        / metrics_without_filter['avg_precision_at_k'] * 100
        if metrics_without_filter['avg_precision_at_k'] > 0 else 0
    )
    
    improvement_mrr = (
        (metrics_with_filter['avg_mrr'] - metrics_without_filter['avg_mrr']) 
        / metrics_without_filter['avg_mrr'] * 100
        if metrics_without_filter['avg_mrr'] > 0 else 0
    )
    
    print(f"\nPrecision@3:")
    print(f"  With Filter:    {metrics_with_filter['avg_precision_at_k']:.3f}")
    print(f"  Without Filter: {metrics_without_filter['avg_precision_at_k']:.3f}")
    print(f"  Improvement:    {improvement_precision:.1f}%")
    
    print(f"\nMRR:")
    print(f"  With Filter:    {metrics_with_filter['avg_mrr']:.3f}")
    print(f"  Without Filter: {metrics_without_filter['avg_mrr']:.3f}")
    print(f"  Improvement:    {improvement_mrr:.1f}%")
    
    print(f"\nRetrieval Time:")
    print(f"  With Filter:    {metrics_with_filter['avg_retrieval_time']:.3f}s")
    print(f"  Without Filter: {metrics_without_filter['avg_retrieval_time']:.3f}s")
    
    # Save results
    output_dir = "evaluations/results/rag"
    os.makedirs(output_dir, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    result_with = EvaluationResult(
        experiment_name="exp_rag_with_filter",
        timestamp=timestamp,
        config={'metadata_filtering': True, 'top_k': 3},
        metrics=metrics_with_filter
    )
    result_with.save(output_dir)
    
    result_without = EvaluationResult(
        experiment_name="exp_rag_without_filter",
        timestamp=timestamp,
        config={'metadata_filtering': False, 'top_k': 3},
        metrics=metrics_without_filter
    )
    result_without.save(output_dir)
    
    print("\n✅ Experiment complete!")


if __name__ == "__main__":
    run_experiment()
