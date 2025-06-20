# PyBind11 Object-Oriented Hello World Project

This project demonstrates how to use pybind11 to expose C++ classes to Python, creating a truly object-oriented API that maintains state and provides rich functionality.

## Project Structure

```
kvtrans/
├── hello.h           # Header file with Greeter class declaration
├── hello.cc          # C++ implementation file with Greeter class methods
├── pybind_hello.cc   # pybind11 wrapper code to expose the class
├── Makefile          # Build configuration
├── test_hello.py     # Python test script demonstrating OOP usage
└── README.md         # This file
```

## Key Features

### Object-Oriented Design
- **C++ Class**: `Greeter` class with private state and public methods
- **State Management**: Maintains greeting history and counts
- **Encapsulation**: Private member variables with controlled access
- **Python Integration**: Seamlessly exposes C++ objects to Python

### Rich Functionality
- **Customizable Greetings**: Default and custom greeting messages
- **History Tracking**: Keeps track of all greetings performed
- **Mathematical Operations**: Basic arithmetic with different data types
- **State Inspection**: Methods to query and modify internal state

## Prerequisites

- Python 3.x with development headers
- pybind11 library
- C++ compiler (g++) with C++11 support

## Installation

1. Install pybind11 if not already installed:
   ```bash
   make install-deps
   ```

2. Build the module:
   ```bash
   make
   ```

3. Run the comprehensive tests:
   ```bash
   make test
   ```

## Usage Examples

### Basic Usage

```python
import hello_module

# Create a greeter with default greeting
greeter = hello_module.Greeter()
print(greeter)  # Shows object representation

# Create a greeter with custom greeting
custom_greeter = hello_module.Greeter("Howdy")

# Greet someone
greeting = greeter.greet("World")
print(greeting)  # Output: Hello, World! Welcome to pybind11 OOP!

# Use custom greeting for specific call
special_greeting = greeter.greet("Developer", "Welcome")
print(special_greeting)  # Output: Welcome, Developer! Welcome to pybind11 OOP!
```

### State Management

```python
# Check greeting statistics
count = greeter.get_greeting_count()
history = greeter.get_greeting_history()

# Modify default greeting
greeter.set_default_greeting("Hi there")
current_greeting = greeter.get_default_greeting()

# Get comprehensive summary
summary = greeter.get_summary()
print(summary)

# Clear history
greeter.clear_history()
```

### Mathematical Operations

```python
# Integer arithmetic
result = greeter.add_numbers(5, 3)
print(result)  # Output: 8

# Floating-point arithmetic
product = greeter.multiply_numbers(2.5, 4.0)
print(product)  # Output: 10.0
```

### Multiple Independent Objects

```python
# Create multiple greeters with different personalities
formal_greeter = hello_module.Greeter("Good day")
casual_greeter = hello_module.Greeter("Hey")

# Each maintains its own state
formal_greeter.greet("Professor")
casual_greeter.greet("Friend")

# Independent greeting counts
print(f"Formal: {formal_greeter.get_greeting_count()}")
print(f"Casual: {casual_greeter.get_greeting_count()}")
```

## Available Make Targets

- `make all` (default) - Build the pybind11 module
- `make clean` - Remove build artifacts
- `make test` - Run the comprehensive OOP test suite
- `make install-deps` - Install pybind11 dependency
- `make help` - Show available targets

## Greeter Class API

### Constructor
- `Greeter(default_greeting="Hello")` - Create a new greeter with optional default greeting

### Greeting Methods
- `greet(name, custom_greeting="")` - Greet someone with default or custom greeting
- `get_greeting_count()` - Get the total number of greetings performed
- `get_greeting_history()` - Get list of all greetings performed

### Configuration Methods
- `set_default_greeting(new_greeting)` - Change the default greeting
- `get_default_greeting()` - Get the current default greeting
- `clear_history()` - Reset greeting history and count

### Mathematical Methods
- `add_numbers(a, b)` - Add two integers
- `multiply_numbers(a, b)` - Multiply two floating-point numbers

### Utility Methods
- `get_summary()` - Get a formatted summary of the greeter's state

## Object-Oriented Benefits

1. **Encapsulation**: Internal state is protected and accessed through controlled methods
2. **State Persistence**: Each greeter instance maintains its own history and configuration
3. **Reusability**: Multiple greeter instances can coexist with different behaviors
4. **Extensibility**: Easy to add new methods and functionality to the class
5. **Pythonic**: Feels natural to Python developers while leveraging C++ performance

## Design Patterns Demonstrated

- **Constructor Overloading**: Default and custom initialization
- **Method Overloading**: Methods with optional parameters
- **State Management**: Private members with public accessors
- **RAII**: Proper resource management through constructor/destructor
- **Const Correctness**: Read-only methods marked as const 