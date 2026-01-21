import json
import time
from typing import List, Dict, Tuple
from datetime import datetime
import os

class Evaluator:
    """
    Evaluation framework for RAG system.
    Processes eval.jsonl and calculates metrics.
    """
    
    def __init__(self, eval_file: str = "eval/eval.jsonl"):
        self.eval_file = eval_file
        self.results = []
    
    def load_eval_data(self) -> List[Dict]:
        """Load evaluation dataset from JSONL file."""
        if not os.path.exists(self.eval_file):
            return []
        
        eval_data = []
        with open(self.eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))
        
        return eval_data
    
    def evaluate_query(self, 
                      query: str, 
                      expected_docs: List[str],
                      retrieved_chunks: List[Dict],
                      answer: str,
                      latency: float) -> Dict:
        """
        Evaluate a single query.
        
        Args:
            query: The question
            expected_docs: List of expected document names/keywords
            retrieved_chunks: Retrieved chunks with 'filename' field
            answer: Generated answer
            latency: Time taken in seconds
            
        Returns:
            Evaluation result dictionary
        """
        # Check if any expected docs are in retrieved results
        retrieved_docs = [chunk['filename'] for chunk in retrieved_chunks]
        
        hits = []
        for expected in expected_docs:
            for retrieved in retrieved_docs:
                if expected.lower() in retrieved.lower():
                    hits.append(expected)
                    break
        
        hit_rate = len(hits) / len(expected_docs) if expected_docs else 0
        
        # Check if citations exist
        has_citations = "【引用】" in answer or "文档" in answer or "[" in answer
        
        return {
            'query': query,
            'hit_rate': hit_rate,
            'hits': hits,
            'expected': expected_docs,
            'retrieved_count': len(retrieved_chunks),
            'has_citations': has_citations,
            'latency': latency,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate aggregate metrics from evaluation results.
        
        Returns:
            Dictionary of metrics
        """
        if not results:
            return {
                'avg_hit_rate': 0,
                'citation_rate': 0,
                'avg_latency': 0,
                'total_queries': 0
            }
        
        total_hit_rate = sum(r['hit_rate'] for r in results)
        total_citations = sum(1 for r in results if r['has_citations'])
        total_latency = sum(r['latency'] for r in results)
        
        return {
            'avg_hit_rate': total_hit_rate / len(results),
            'citation_rate': total_citations / len(results),
            'avg_latency': total_latency / len(results),
            'total_queries': len(results),
            'results': results
        }
    
    def export_report(self, metrics: Dict, output_file: str = "eval/report.json"):
        """Export evaluation report to JSON."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(f"Report exported to {output_file}")
    
    def export_csv(self, results: List[Dict], output_file: str = "eval/report.csv"):
        """Export evaluation results to CSV."""
        import csv
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'query', 'hit_rate', 'hits', 'expected', 'retrieved_count', 
                'has_citations', 'latency', 'timestamp'
            ])
            writer.writeheader()
            
            for result in results:
                # Convert lists to strings for CSV
                row = result.copy()
                row['hits'] = ', '.join(row['hits'])
                row['expected'] = ', '.join(row['expected'])
                writer.writerow(row)
        
        print(f"CSV exported to {output_file}")
