#!/usr/bin/env bash

# XXX: Do not invoke separately. Use `make doc` (or used by the CI).

MAP_FILE="libdivERGe.map"
SO_FILE="libdivERGe.so"

if [ ! -f "$MAP_FILE" ]; then
    echo "Error: File '$MAP_FILE' not found." >&2
    exit 1
fi

if [ -f "$SO_FILE" ]; then
    echo "shared object exists, not building mock."
    exit 0
fi

awk '
    # Detect when we enter a global block
    /global:/ { in_global = 1; next }

    # Detect when we enter a local block or leave the namespace
    /local:/ || /};/ { in_global = 0; next }

    # If we are inside a global block, clean up and print the function names
    in_global {
        # Remove leading/trailing whitespace
        gsub(/^[[:space:]]+|[[:space:]]+$/, "")

        # Strip the trailing semicolon if it exists
        sub(/;$/, "")

        # Only print if the line is not empty and not a wildcard
        if ($0 != "" && $0 != "*") {
            print "void", $0, "( void ) {}"
        }
    }
' "$MAP_FILE" | gcc -x c - -shared -o "$SO_FILE"

