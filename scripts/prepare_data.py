import json
import argparse
from pathlib import Path
import logging
from typing import List, Dict
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset(output_dir: Path, num_samples: int = 1000):
    """Create sample dataset for demonstration"""
    
    python_samples = [
        {
            'code': '''def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1''',
            'docstring': 'Binary search algorithm to find target in sorted array',
            'language': 'python',
            'func_name': 'binary_search'
        },
        {
            'code': '''def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result''',
            'docstring': 'Merge sort algorithm for sorting arrays',
            'language': 'python',
            'func_name': 'merge_sort'
        },
        {
            'code': '''def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b''',
            'docstring': 'Calculate nth Fibonacci number',
            'language': 'python',
            'func_name': 'fibonacci'
        }
    ]
    
    javascript_samples = [
        {
            'code': '''function quickSort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[0];
    const left = arr.slice(1).filter(x => x < pivot);
    const right = arr.slice(1).filter(x => x >= pivot);
    return [...quickSort(left), pivot, ...quickSort(right)];
}''',
            'docstring': 'Quick sort algorithm implementation',
            'language': 'javascript',
            'func_name': 'quickSort'
        },
        {
            'code': '''function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}''',
            'docstring': 'Debounce function to limit rate of function calls',
            'language': 'javascript',
            'func_name': 'debounce'
        }
    ]
    
    java_samples = [
        {
            'code': '''public class BubbleSort {
    public static void sort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n-1; i++) {
            for (int j = 0; j < n-i-1; j++) {
                if (arr[j] > arr[j+1]) {
                    int temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
        }
    }
}''',
            'docstring': 'Bubble sort algorithm for sorting arrays',
            'language': 'java',
            'func_name': 'sort'
        }
    ]
    
    all_samples = python_samples + javascript_samples + java_samples
    
    dataset = []
    for _ in range(num_samples):
        sample = random.choice(all_samples).copy()
        dataset.append(sample)
    
    random.shuffle(dataset)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / 'val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(output_dir / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Created dataset with {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--output_dir', type=str, default='./data/datasets',
                        help='Output directory for prepared data')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    logger.info("Creating sample dataset...")
    create_sample_dataset(output_dir, args.num_samples)
    
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()