#!/usr/bin/env python3
"""
Test script for the hello_module pybind11 extension with object-oriented API
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

def test_greeter_creation():
    """Test creating Greeter objects"""
    print("\n=== Testing Greeter Creation ===")
    
    # Default greeter
    greeter1 = hello_module.Greeter()
    print(f"Default greeter: {greeter1}")
    print(f"Default greeting: '{greeter1.get_default_greeting()}'")
    
    # Custom greeter
    greeter2 = hello_module.Greeter("Hi there")
    print(f"Custom greeter: {greeter2}")
    print(f"Custom greeting: '{greeter2.get_default_greeting()}'")
    
    return greeter1, greeter2

def test_greeting_functionality(greeter):
    """Test the greeting methods"""
    print(f"\n=== Testing Greeting Functionality ===")
    
    test_names = ["World", "Python", "C++", "pybind11"]
    
    for name in test_names:
        result = greeter.greet(name)
        print(f"greet('{name}') = '{result}'")
    
    # Test custom greeting
    custom_result = greeter.greet("Developer", "Welcome")
    print(f"greet('Developer', 'Welcome') = '{custom_result}'")
    
    # Check greeting count
    count = greeter.get_greeting_count()
    print(f"Total greetings: {count}")

def test_math_operations(greeter):
    """Test mathematical operations"""
    print(f"\n=== Testing Math Operations ===")
    
    # Test addition
    test_cases_add = [(1, 2), (10, 20), (-5, 15), (0, 0)]
    for a, b in test_cases_add:
        result = greeter.add_numbers(a, b)
        expected = a + b
        status = "✓" if result == expected else "✗"
        print(f"{status} add_numbers({a}, {b}) = {result} (expected: {expected})")
    
    # Test multiplication
    test_cases_mult = [(2.5, 3.0), (1.5, 2.0), (-2.0, 3.5), (0.0, 5.0)]
    for a, b in test_cases_mult:
        result = greeter.multiply_numbers(a, b)
        expected = a * b
        status = "✓" if abs(result - expected) < 1e-10 else "✗"
        print(f"{status} multiply_numbers({a}, {b}) = {result} (expected: {expected})")

def test_state_management(greeter):
    """Test state management features"""
    print(f"\n=== Testing State Management ===")
    
    # Check initial state
    print(f"Initial greeting count: {greeter.get_greeting_count()}")
    print(f"Initial history length: {len(greeter.get_greeting_history())}")
    
    # Add some greetings
    greeter.greet("Alice")
    greeter.greet("Bob")
    greeter.greet("Charlie")
    
    # Check updated state
    print(f"After 3 greetings:")
    print(f"  Greeting count: {greeter.get_greeting_count()}")
    print(f"  History length: {len(greeter.get_greeting_history())}")
    
    # Show history
    history = greeter.get_greeting_history()
    print("  Recent greetings:")
    for i, greeting in enumerate(history[-3:], 1):
        print(f"    {i}. {greeting}")
    
    # Test changing default greeting
    old_greeting = greeter.get_default_greeting()
    greeter.set_default_greeting("Howdy")
    print(f"Changed default greeting from '{old_greeting}' to '{greeter.get_default_greeting()}'")
    
    # Test with new greeting
    result = greeter.greet("Cowboy")
    print(f"New greeting style: '{result}'")
    
    # Show summary
    print("\nGreeter Summary:")
    print(greeter.get_summary())
    
    # Test clearing history
    greeter.clear_history()
    print(f"\nAfter clearing history:")
    print(f"  Greeting count: {greeter.get_greeting_count()}")
    print(f"  History length: {len(greeter.get_greeting_history())}")

def test_multiple_greeters():
    """Test multiple independent greeter instances"""
    print(f"\n=== Testing Multiple Greeters ===")
    
    # Create different greeters
    formal_greeter = hello_module.Greeter("Good day")
    casual_greeter = hello_module.Greeter("Hey")
    
    # Use them independently
    formal_result = formal_greeter.greet("Professor")
    casual_result = casual_greeter.greet("Buddy")
    
    print(f"Formal greeter: '{formal_result}'")
    print(f"Casual greeter: '{casual_result}'")
    
    # Check they maintain separate state
    print(f"Formal greeter count: {formal_greeter.get_greeting_count()}")
    print(f"Casual greeter count: {casual_greeter.get_greeting_count()}")

def test_module_info():
    """Test module documentation"""
    print("\n=== Module Information ===")
    print(f"Module docstring: {hello_module.__doc__}")
    print(f"Greeter class available: {'Greeter' in dir(hello_module)}")
    
    # Create a greeter to test its methods
    greeter = hello_module.Greeter()
    available_methods = [method for method in dir(greeter) if not method.startswith('_')]
    print(f"Available Greeter methods: {available_methods}")

def main():
    """Run all tests"""
    print("Running pybind11 hello_module OOP tests...")
    
    test_module_info()
    
    # Create test greeters
    default_greeter, custom_greeter = test_greeter_creation()
    
    # Test functionality with default greeter
    test_greeting_functionality(default_greeter)
    test_math_operations(default_greeter)
    test_state_management(default_greeter)
    
    # Test multiple greeters
    test_multiple_greeters()
    
    print("\n=== All OOP tests completed! ===")

if __name__ == "__main__":
    main() 