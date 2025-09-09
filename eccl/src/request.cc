#include "request.h"

std::atomic<unsigned> Request::global_id_counter{0};

void Request::onCommCompletion() {}

void Request::onComputeCompletion() {}