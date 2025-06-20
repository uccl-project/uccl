# PyBind11 Hello World Project

This is a simple demonstration of using pybind11 to expose C++ functions to Python.

## Project Structure

```
kvtrans/
├── hello.h           # Header file with function declarations
├── hello.cc          # C++ implementation file
├── pybind_hello.cc   # pybind11 wrapper code
├── Makefile          # Build configuration
├── test_hello.py     # Python test script
└── README.md         # This file
```

## Prerequisites

- Python 3.x with development headers
- pybind11 library
- C++ compiler (g++)

## Installation

1. Install pybind11 if not already installed:
   ```bash
   make install-deps
   ```

2. Build the module:
   ```bash
   make
   ```

3. Run the tests:
   ```bash
   make test
   ```

## Manual Usage

After building, you can use the module in Python:

```python
import hello_module

# Test the hello world function
greeting = hello_module.hello_world("Python")
print(greeting)  # Output: Hello, Python! Welcome to pybind11!

# Test the add numbers function
result = hello_module.add_numbers(5, 3)
print(result)  # Output: 8
```

## Available Make Targets

- `make all` (default) - Build the pybind11 module
- `make clean` - Remove build artifacts
- `make test` - Run the test script
- `make install-deps` - Install pybind11 dependency
- `make help` - Show available targets

## Functions

### `hello_world(name: str) -> str`
Returns a greeting message for the given name.

### `add_numbers(a: int, b: int) -> int`
Returns the sum of two integers. 