#!/usr/bin/env python3
"""
Test script for the hello_module pybind11 extension
"""

import sys
import os

# Add current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import hello_module
    print("✓ Successfully imported hello_module")
except ImportError as e:
    print(f"✗ Failed to import hello_module: {e}")
    print("Make sure to run 'make' first to build the module")
    sys.exit(1)

def test_hello_world():
    """Test the hello_world function"""
    print("\n=== Testing hello_world function ===")
    
    test_cases = ["World", "pybind11", "C++", "Python"]
    
    for name in test_cases:
        result = hello_module.hello_world(name)
        print(f"hello_world('{name}') = '{result}'")

def test_add_numbers():
    """Test the add_numbers function"""
    print("\n=== Testing add_numbers function ===")
    
    test_cases = [(1, 2), (10, 20), (-5, 15), (0, 0)]
    
    for a, b in test_cases:
        result = hello_module.add_numbers(a, b)
        expected = a + b
        status = "✓" if result == expected else "✗"
        print(f"{status} add_numbers({a}, {b}) = {result} (expected: {expected})")

def test_module_info():
    """Test module documentation"""
    print("\n=== Module Information ===")
    print(f"Module docstring: {hello_module.__doc__}")
    print(f"Available functions: {[name for name in dir(hello_module) if not name.startswith('_')]}")

def main():
    """Run all tests"""
    print("Running pybind11 hello_module tests...")
    
    test_module_info()
    test_hello_world()
    test_add_numbers()
    
    print("\n=== All tests completed! ===")

if __name__ == "__main__":
    main() 