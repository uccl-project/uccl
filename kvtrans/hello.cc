#include "hello.h"
#include <iomanip>
#include <sstream>

// Constructor
Greeter::Greeter(std::string const& default_greeting)
    : default_greeting_(default_greeting), greeting_count_(0) {}

// Destructor
Greeter::~Greeter() {
  // Clean up if needed (though not necessary for this simple example)
}

// Greet method
std::string Greeter::greet(std::string const& name,
                           std::string const& custom_greeting) {
  std::ostringstream oss;
  std::string greeting_to_use =
      custom_greeting.empty() ? default_greeting_ : custom_greeting;

  oss << greeting_to_use << ", " << name << "! Welcome to pybind11 OOP!";
  std::string result = oss.str();

  // Store in history and increment counter
  greeting_history_.push_back(result);
  greeting_count_++;

  return result;
}

// Add numbers method
int Greeter::add_numbers(int a, int b) { return a + b; }

// Multiply numbers method
double Greeter::multiply_numbers(double a, double b) { return a * b; }

// Get greeting count
int Greeter::get_greeting_count() const { return greeting_count_; }

// Get greeting history
std::vector<std::string> const& Greeter::get_greeting_history() const {
  return greeting_history_;
}

// Set default greeting
void Greeter::set_default_greeting(std::string const& new_greeting) {
  default_greeting_ = new_greeting;
}

// Get default greeting
std::string const& Greeter::get_default_greeting() const {
  return default_greeting_;
}

// Clear history
void Greeter::clear_history() {
  greeting_history_.clear();
  greeting_count_ = 0;
}

// Get summary
std::string Greeter::get_summary() const {
  std::ostringstream oss;
  oss << "Greeter Summary:\n";
  oss << "  Default greeting: \"" << default_greeting_ << "\"\n";
  oss << "  Total greetings: " << greeting_count_ << "\n";
  oss << "  History size: " << greeting_history_.size();
  return oss.str();
}