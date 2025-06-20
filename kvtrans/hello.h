#ifndef HELLO_H
#define HELLO_H

#include <string>
#include <vector>

/**
 * A simple Greeter class that demonstrates object-oriented programming
 * with pybind11 integration
 */
class Greeter {
 private:
  std::string default_greeting_;
  std::vector<std::string> greeting_history_;
  int greeting_count_;

 public:
  /**
   * Constructor with optional default greeting
   * @param default_greeting The default greeting prefix (default: "Hello")
   */
  explicit Greeter(std::string const& default_greeting = "Hello");

  /**
   * Destructor
   */
  ~Greeter();

  /**
   * Greet someone with the default or custom greeting
   * @param name The name to greet
   * @param custom_greeting Optional custom greeting (overrides default)
   * @return A greeting string
   */
  std::string greet(std::string const& name,
                    std::string const& custom_greeting = "");

  /**
   * Add two numbers
   * @param a First number
   * @param b Second number
   * @return Sum of a and b
   */
  int add_numbers(int a, int b);

  /**
   * Multiply two numbers
   * @param a First number
   * @param b Second number
   * @return Product of a and b
   */
  double multiply_numbers(double a, double b);

  /**
   * Get the number of greetings performed
   * @return Number of greetings
   */
  int get_greeting_count() const;

  /**
   * Get the greeting history
   * @return Vector of all greetings performed
   */
  std::vector<std::string> const& get_greeting_history() const;

  /**
   * Set a new default greeting
   * @param new_greeting The new default greeting
   */
  void set_default_greeting(std::string const& new_greeting);

  /**
   * Get the current default greeting
   * @return The current default greeting
   */
  std::string const& get_default_greeting() const;

  /**
   * Clear the greeting history
   */
  void clear_history();

  /**
   * Get a summary of the greeter's state
   * @return A string describing the greeter's current state
   */
  std::string get_summary() const;
};

#endif  // HELLO_H