# !/bin/bash

get_nodes() {
    local input="$1"  # Path to the file containing the input
    awk '!/^#|^$/ {print $0}' "$input" | paste -sd "," -
}