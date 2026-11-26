# Pure CPU tests; deliberately independent of GPU SDK/profile discovery.
CXX ?= c++
PYTHON ?= python3
CPPFLAGS += -I..
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++17 -Wall -Wextra -Werror -Wno-missing-field-initializers
BUILD_DIR ?= .build-host

.PHONY: test sanitize clean
test: $(BUILD_DIR)/test_proxy_scheduler
	$(BUILD_DIR)/test_proxy_scheduler
	CXX='$(CXX)' CXXFLAGS='$(CXXFLAGS)' LDFLAGS='$(LDFLAGS)' $(PYTHON) test_cooperative_proxy_adapter.py
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) test_proxy_fairness_benchmark.py

$(BUILD_DIR)/test_proxy_scheduler: test_proxy_scheduler.cc ../util/proxy_scheduler.h
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

sanitize:
	$(MAKE) test BUILD_DIR=$(BUILD_DIR)/sanitize \
	  CXXFLAGS='-O1 -g -std=c++17 -Wall -Wextra -Werror -Wno-missing-field-initializers -fsanitize=address,undefined -fno-omit-frame-pointer' \
	  LDFLAGS='-fsanitize=address,undefined'

clean:
	rm -rf $(BUILD_DIR)
