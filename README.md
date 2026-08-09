# Leaky Integrate-and-Fire Neuron on FPGA

A digital implementation of the Leaky Integrate-and-Fire (LIF) spiking neuron model written in Verilog, targeting the Basys 3 FPGA board.

## What This Is

The LIF model is one of the simplest spiking neuron models used in computational neuroscience. It describes a neuron's membrane potential that integrates incoming signals, leaks over time, and fires a spike when it crosses a threshold. After firing, the neuron resets and enters a refractory period where it ignores all input.

This project implements that behavior in synthesizable Verilog with three synaptic inputs, configurable weights, and a tunable decay rate.

## How It Works

The neuron operates as a four-state machine:

1. **Integrate** - Applies exponential decay to the current membrane potential and adds any weighted input spikes. The decay is approximated using a right-shift by `alpha` bits, so the potential each cycle becomes `V * (1 - 1/2^alpha) + weighted_inputs`.
2. **Spike** - Asserts the output spike signal for one clock cycle.
3. **Reset** - Clears the output signal and resets the membrane potential to zero.
4. **Refractory** - Holds for 10 clock cycles, ignoring all input before returning to the integrate state.

The threshold is set to 100. The decay approximation comes from a discretized form of the LIF differential equation where the exponential term `e^(-t/tau)` is replaced by a bit-shift operation for hardware efficiency.

## Files

- `lif_neuron.v` - Neuron module
- `lif_neuron_tb.v` - Testbench with three synaptic inputs cycling through various spike patterns
- `output.txt` - Simulation output log
- `PROJECT-TIMELINE.md` - Notes on the learning process and derivation of the math

## Running the Simulation

Open the testbench in Vivado or any Verilog simulator. The testbench uses `$monitor` to log signal changes to the console. The provided `output.txt` shows the membrane potential building up across input spikes, firing once at threshold, resetting to zero, and then decaying quietly through the refractory period.

## Background

The math behind the decay approximation is derived from the LIF differential equation and documented in `PROJECT-TIMELINE.md` with handwritten notes. The key insight is that the exponential decay factor can be replaced with a power-of-two division (bit shift) to avoid any multipliers or dividers in hardware.
