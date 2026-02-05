"""
Simple test script to verify external judge functionality.

This script demonstrates how to test an external judge before using it in the full pipeline.
"""

import pandas as pd
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clear_eval.pipeline.external_judge import load_external_judge, call_external_judge
from src.clear_eval.pipeline.constants import EVALUATION_TEXT_COL, SCORE_COL


def test_exact_match_judge():
    """Test the exact match judge with sample data."""
    print("=" * 60)
    print("Testing Exact Match Judge")
    print("=" * 60)
    
    # Load the judge
    judge_path = os.path.join(os.path.dirname(__file__), 'custom_judges', 'exact_match_judge.py')
    judge_func = load_external_judge(judge_path, 'evaluate')
    print(f"✓ Successfully loaded judge from: {judge_path}\n")
    
    # Create test data
    test_df = pd.DataFrame([
        {
            'id': 'test_1',
            'model_input': 'What is 2+2?',
            'response': '4',
            'ground_truth': '4',
            'expected_score': 1.0
        },
        {
            'id': 'test_2',
            'model_input': 'What is the capital of France?',
            'response': 'Paris',
            'ground_truth': 'paris',  # Different case
            'expected_score': 1.0  # Should match (case-insensitive)
        },
        {
            'id': 'test_3',
            'model_input': 'What is 5*5?',
            'response': '24',
            'ground_truth': '25',
            'expected_score': 0.0
        }
    ])
    
    config = {
        'model_output_column': 'response',
        'reference_column': 'ground_truth',
        'external_judge_config': {}
    }
    
    # Call judge with entire DataFrame
    result_df = call_external_judge(judge_func, test_df, config)
    
    # Test each result
    all_passed = True
    for idx, row in result_df.iterrows():
        expected_score = row['expected_score']
        actual_score = row[SCORE_COL]
        eval_text = row[EVALUATION_TEXT_COL]
        
        passed = actual_score == expected_score
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"{status} | Test: {row['id']}")
        print(f"  Response: '{row['response']}'")
        print(f"  Ground Truth: '{row['ground_truth']}'")
        print(f"  Score: {actual_score} (expected: {expected_score})")
        print(f"  Evaluation: {eval_text[:80]}...")
        print()
    
    return all_passed


def test_numeric_tolerance_judge():
    """Test the numeric tolerance judge with sample data."""
    print("=" * 60)
    print("Testing Numeric Tolerance Judge")
    print("=" * 60)
    
    # Load the judge
    judge_path = os.path.join(os.path.dirname(__file__), 'custom_judges', 'numeric_tolerance_judge.py')
    judge_func = load_external_judge(judge_path, 'evaluate')
    print(f"✓ Successfully loaded judge from: {judge_path}\n")
    
    # Create test data
    test_df = pd.DataFrame([
        {
            'id': 'test_1',
            'model_input': 'What is 10 * 10?',
            'response': '100',
            'ground_truth': '100',
            'expected_score': 1.0
        },
        {
            'id': 'test_2',
            'model_input': 'Calculate 1/3',
            'response': '0.333',
            'ground_truth': '0.3333',
            'expected_score': 1.0  # Within 1% tolerance
        },
        {
            'id': 'test_3',
            'model_input': 'What is 50 + 50?',
            'response': '95',
            'ground_truth': '100',
            'expected_score': 0.0  # 5% error, exceeds 1% tolerance
        }
    ])
    
    config = {
        'model_output_column': 'response',
        'reference_column': 'ground_truth',
        'external_judge_config': {
            'tolerance': 0.01,  # 1% tolerance
            'extract_last_number': False
        }
    }
    
    # Call judge with entire DataFrame
    result_df = call_external_judge(judge_func, test_df, config)
    
    # Test each result
    all_passed = True
    for idx, row in result_df.iterrows():
        expected_score = row['expected_score']
        actual_score = row[SCORE_COL]
        eval_text = row[EVALUATION_TEXT_COL]
        
        passed = actual_score == expected_score
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"{status} | Test: {row['id']}")
        print(f"  Response: '{row['response']}'")
        print(f"  Ground Truth: '{row['ground_truth']}'")
        print(f"  Score: {actual_score} (expected: {expected_score})")
        print(f"  Evaluation: {eval_text[:80]}...")
        print()
    
    return all_passed


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CLEAR External Judge Test Suite")
    print("=" * 60 + "\n")
    
    try:
        # Test exact match judge
        exact_match_passed = test_exact_match_judge()
        
        # Test numeric tolerance judge
        numeric_tolerance_passed = test_numeric_tolerance_judge()
        
        # Summary
        print("=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Exact Match Judge: {'✓ PASSED' if exact_match_passed else '✗ FAILED'}")
        print(f"Numeric Tolerance Judge: {'✓ PASSED' if numeric_tolerance_passed else '✗ FAILED'}")
        
        if exact_match_passed and numeric_tolerance_passed:
            print("\n✓ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

