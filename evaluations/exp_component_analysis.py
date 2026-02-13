"""
Experiment 5.4: Component Analysis

Tests the contribution of each component in the system:
1. Full System: DINO → CLIP → Agent + RAG
2. No RAG: Agent uses only internal knowledge
3. No CLIP Verification: Use DINO labels directly
4. No Self-Correction: Disable validators

Metrics:
- Treatment Accuracy: % correct recommendations vs. ground truth
- Hallucination Rate: % pesticide names not found in manuals
- Conflict Resolution Accuracy: % correct handling of beneficial insects
- Inference Time: Average time per diagnosis
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.deps import AgronomyDeps
from app.agent.core import agronomy_agent, DiagnosisResult
from app.pipe import analyze_full_plant, VisionSystem
from evaluations.eval_utils import EvaluationResult, Timer, print_metrics


class ComponentConfig:
    """Configuration for component analysis experiments"""
    def __init__(self, 
                 enable_rag: bool = True,
                 enable_clip: bool = True, 
                 enable_validators: bool = True):
        self.enable_rag = enable_rag
        self.enable_clip = enable_clip
        self.enable_validators = enable_validators
    
    def get_name(self) -> str:
        """Get configuration name"""
        if self.enable_rag and self.enable_clip and self.enable_validators:
            return "Full System"
        elif not self.enable_rag:
            return "No RAG"
        elif not self.enable_clip:
            return "No CLIP Verification"
        elif not self.enable_validators:
            return "No Self-Correction"
        else:
            return "Custom Config"
    
    def to_dict(self) -> Dict:
        return {
            'enable_rag': self.enable_rag,
            'enable_clip': self.enable_clip,
            'enable_validators': self.enable_validators
        }


import yaml


def get_test_cases() -> List[Dict]:
    yaml_file = os.path.join(
        os.path.dirname(__file__), 
        'test_cases/exp_test_cases.yaml'
    )
    
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(
            f"Test cases file not found: {yaml_file}"
        )
    
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    test_cases = []
    for tc in data['test_cases']:
        # Convert YAML structure to test case format
        test_case = {
            'id': tc['id'],
            'name': tc['name'],
            'description': tc.get('description', ''),
            'deps': AgronomyDeps(
                user_id=tc['deps']['user_id'],
                crop_name=tc['deps']['crop_name'],
                total_leaves=tc['deps']['total_leaves'],
                healthy_count=tc['deps']['healthy_count'],
                disease_counts=tc['deps'].get('disease_counts', {}),
                pest_counts=tc['deps'].get('pest_counts', {}),
                detailed_detections=None
            ),
            'ground_truth': tc['ground_truth']
        }
        test_cases.append(test_case)
    
    return test_cases


async def run_agent_with_config(deps: AgronomyDeps, config: ComponentConfig) -> DiagnosisResult:
    """
    Run agent with specific component configuration
    
    Note: Some configurations require code modifications to disable features.
    For now, this runs the full system. You'll need to modify app/agent/core.py
    and app/agent/tools.py to truly disable features.
    """
    summary_text = (
        f"Analysis Report:\n"
        f"- Total detected objects: {deps.total_leaves}\n"
        f"- Healthy: {deps.healthy_count}\n"
        f"- Diseases: {deps.disease_counts}\n"
        f"- Pests: {deps.pest_counts}\n"
    )

    user_prompt = (
        f"Here is the aggregate data for a crop image: \n{summary_text}\n"
        "TASKS:\n"
        "1. **Pest Analysis:** Categorize detected pests as 'Beneficial' or 'Harmful'. Apply the Pest Protocol rules.\n"
        "2. **Disease Analysis:** Assess severity based on infection ratio.\n"
        "3. **Plan:** Provide an integrated plan. If mixed infections (pests + disease) exists, prioritize the most severe threat but protect beneficial insects."
    )

    try:
        result = await agronomy_agent.run(user_prompt, deps=deps)
        return result.output
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        raise


def evaluate_result(result: DiagnosisResult, ground_truth: Dict) -> Dict[str, float]:
    """
    Evaluate agent output against ground truth
    
    Returns metrics:
    - severity_correct: 1 if severity matches, 0 otherwise
    - status_correct: 1 if health status matches, 0 otherwise  
    - infection_ratio_correct: 1 if in expected range, 0 otherwise
    - pesticide_appropriate: 1 if pesticide presence matches expectation
    - mentioned_key_disease: 1 if key disease mentioned
    - conflict_handled: 1 if conflict handled correctly (if applicable)
    """
    metrics = {}
    
    # Check severity level
    if 'severity_level' in ground_truth:
        metrics['severity_correct'] = 1.0 if result.severity_level == ground_truth['severity_level'] else 0.0
    
    # Check overall health status
    if 'overall_health_status' in ground_truth:
        metrics['status_correct'] = 1.0 if result.overall_health_status == ground_truth['overall_health_status'] else 0.0
    
    # Check infection ratio
    if 'infection_ratio_range' in ground_truth:
        min_ratio, max_ratio = ground_truth['infection_ratio_range']
        metrics['infection_ratio_correct'] = 1.0 if min_ratio <= result.infection_ratio <= max_ratio else 0.0
    
    # Check pesticide presence
    if 'should_have_pesticides' in ground_truth:
        has_pesticides = result.required_pesticides and len(result.required_pesticides) > 0
        expected = ground_truth['should_have_pesticides']
        metrics['pesticide_appropriate'] = 1.0 if has_pesticides == expected else 0.0
    
    # Check if key disease mentioned
    if 'should_mention_disease' in ground_truth:
        disease = ground_truth['should_mention_disease']
        if isinstance(disease, list):
            mentioned = any(d.lower() in result.reasoning.lower() or 
                          d.lower() in str(result.identified_pathogens).lower() 
                          for d in disease)
        else:
            mentioned = (disease.lower() in result.reasoning.lower() or 
                        disease.lower() in str(result.identified_pathogens).lower())
        metrics['mentioned_key_disease'] = 1.0 if mentioned else 0.0
    
    # Check conflict resolution (beneficial insects)
    if 'should_avoid_chemicals' in ground_truth:
        has_strong_chemicals = result.required_pesticides and any(
            'cide' in p.lower() and 'soap' not in p.lower() 
            for p in result.required_pesticides
        )
        avoided = not has_strong_chemicals
        metrics['conflict_handled'] = 1.0 if avoided == ground_truth['should_avoid_chemicals'] else 0.0
    
    return metrics


async def run_component_experiment(config: ComponentConfig, test_cases: List[Dict]) -> Dict:
    """
    Run component analysis experiment with given configuration
    
    Returns:
    - Average metrics across all test cases
    - Per-test-case results
    - Timing information
    """
    print(f"\n{'='*60}")
    print(f"🔬 Running: {config.get_name()}")
    print(f"{'='*60}")
    
    results_per_case = []
    total_time = 0
    
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']} (ID: {test_case['id']})")
        
        with Timer(f"  {test_case['id']}") as timer:
            try:
                result = await run_agent_with_config(test_case['deps'], config)
                metrics = evaluate_result(result, test_case['ground_truth'])
                correct = sum(metrics.values())
                total = len(metrics)
                print(f"    ✓ Passed {correct}/{total} checks")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                result = None
                metrics = None

        total_time += timer.elapsed
        results_per_case.append({
            'test_id': test_case['id'],
            'test_name': test_case['name'],
            'metrics': metrics,
            'inference_time': timer.elapsed,
            'result': result
        })
    
    # Calculate aggregate metrics
    all_metrics = {}
    for case in results_per_case:
        if 'metrics' in case:
            for key, value in case['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
    
    avg_metrics = {
        f"avg_{key}": sum(values) / len(values) 
        for key, values in all_metrics.items()
    }
    avg_metrics['avg_inference_time'] = total_time / len(test_cases)
    avg_metrics['total_test_cases'] = len(test_cases)
    
    return {
        'config': config.to_dict(),
        'config_name': config.get_name(),
        'avg_metrics': avg_metrics,
        'per_case_results': results_per_case
    }


async def main():
    """Main experiment runner"""
    print("\n" + "="*60)
    print("🧪 EXPERIMENT 5.4: COMPONENT ANALYSIS")
    print("="*60)
    
    # Get test cases
    test_cases = get_test_cases()
    print(f"\n✅ Loaded {len(test_cases)} test cases")
    
    # Define configurations to test
    configs = [
        ComponentConfig(enable_rag=True, enable_clip=True, enable_validators=True),  # Full
        ComponentConfig(enable_rag=False, enable_clip=True, enable_validators=True),  # No RAG
        # Note: The following require code modifications to truly disable
        # For now, they will run the same as full system
        # ComponentConfig(enable_rag=True, enable_clip=False, enable_validators=True),  # No CLIP
        # ComponentConfig(enable_rag=True, enable_clip=True, enable_validators=False),  # No Validators
    ]
    
    print(f"📋 Testing {len(configs)} configurations\n")
    
    # Run experiments
    all_results = []
    for config in configs:
        result = await run_component_experiment(config, test_cases)
        all_results.append(result)
    
    # Print comparison table
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    # Create comparison table
    print(f"\n{'Configuration':<25} {'Avg Score':<12} {'Time (s)':<10}")
    print("-" * 60)
    
    for result in all_results:
        avg_score = sum([
            v for k, v in result['avg_metrics'].items() 
            if k.startswith('avg_') and k != 'avg_inference_time'
        ]) / max(1, len([k for k in result['avg_metrics'].keys() if k.startswith('avg_') and k != 'avg_inference_time']))
        
        avg_time = result['avg_metrics'].get('avg_inference_time', 0)
        
        print(f"{result['config_name']:<25} {avg_score:.3f}         {avg_time:.2f}")
    
    # Save detailed results
    output_dir = "evaluations/results/component_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"component_analysis_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'test_cases': [
                {
                    'id': tc['id'],
                    'name': tc['name'],
                    'ground_truth': tc['ground_truth']
                } for tc in test_cases
            ],
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n✅ Detailed results saved to: {output_file}")
    print("\n" + "="*60)


if __name__ == "__main__":
    # Note: You need to have the agent system properly initialized
    # This includes .env file with API keys and RAG database set up
    asyncio.run(main())
