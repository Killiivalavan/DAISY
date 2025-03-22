#!/bin/bash
echo "Starting DAISY Voice Assistant..."
echo
echo "Make sure Ollama is running with the llama3.2 model."
echo "If not, open a new terminal and run: ollama serve"
echo

# Check if Ollama is accessible
if ! curl -s --head http://localhost:11434 > /dev/null; then
    echo "Warning: Ollama doesn't seem to be running."
    echo "Starting DAISY anyway, but responses may fail."
    echo
fi

python daisy.py 