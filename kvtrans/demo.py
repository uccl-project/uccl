#!/usr/bin/env python3
"""
Demo script showing the object-oriented pybind11 API
"""

import hello_module

def main():
    print("=== Object-Oriented pybind11 Demo ===\n")
    
    # Create multiple greeter instances with different personalities
    print("Creating different greeter instances...")
    english_greeter = hello_module.Greeter("Hello")
    spanish_greeter = hello_module.Greeter("Hola")
    french_greeter = hello_module.Greeter("Bonjour")
    
    print(f"English: {english_greeter}")
    print(f"Spanish: {spanish_greeter}")
    print(f"French: {french_greeter}")
    
    print("\n=== Multi-lingual Greetings ===")
    print(english_greeter.greet("World"))
    print(spanish_greeter.greet("Mundo"))
    print(french_greeter.greet("Monde"))
    
    print("\n=== State Management ===")
    print(f"English greeter count: {english_greeter.get_greeting_count()}")
    print(f"Spanish greeter count: {spanish_greeter.get_greeting_count()}")
    print(f"French greeter count: {french_greeter.get_greeting_count()}")
    
    print("\n=== Math Operations (with different greeters) ===")
    result1 = english_greeter.add_numbers(10, 20)
    result2 = spanish_greeter.multiply_numbers(3.14, 2.0)
    print(f"Addition: 10 + 20 = {result1}")
    print(f"Multiplication: 3.14 * 2.0 = {result2}")
    
    print("\n=== Dynamic Behavior ===")
    # Change greeting style at runtime
    english_greeter.set_default_greeting("Good morning")
    print(english_greeter.greet("Developer"))
    
    # Show summary
    print(f"\nEnglish greeter summary:")
    print(english_greeter.get_summary())

if __name__ == "__main__":
    main() 