/* D.A.I.S.Y. v2 — AudioWorklet Mic Processor
 *
 * Runs on a dedicated audio thread. Captures 20ms frames of Float32
 * audio from the microphone, converts to Int16, and posts them to
 * the main thread via a zero-copy transferable buffer.
 */

class DaisyMicProcessor extends AudioWorkletProcessor {
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0] || input[0].length === 0) {
            return true; // keep alive, no audio yet
        }

        const floatSamples = input[0]; // Float32Array, typically 128 or 256 samples
        const n = floatSamples.length;
        const int16Samples = new Int16Array(n);

        for (let i = 0; i < n; i++) {
            const s = Math.max(-1.0, Math.min(1.0, floatSamples[i]));
            int16Samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Transfer ownership of the buffer to the main thread (zero-copy)
        this.port.postMessage(int16Samples.buffer, [int16Samples.buffer]);

        return true; // keep processor alive
    }
}

registerProcessor('daisy-mic-processor', DaisyMicProcessor);
