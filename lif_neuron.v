`timescale 1ns / 1ps

module lif_neuron (
    input clk,
    input reset,

    input spike_in_1,
    input spike_in_2,
    input spike_in_3,
    input [7:0] spike_1_weight,
    input [7:0] spike_2_weight,
    input [7:0] spike_3_weight,

    input [7:0] alpha,

    output spike_out
);

    localparam THRESHOLD = 8'd100;

    // alpha is the number of right shift elements and is meant to simulate the decay factor in the differential equation (e^(-t/τ)) where τ is the time constant (R * C) of the neuron

    reg [7:0] Current_Potential = 8'd0;
    reg spike_out_signal;

    localparam [7:0] REFRACTORY_PERIOD = 8'd10; // Number of clock cycles for refractory period
    reg [7:0] refactory_counter = 8'd0;

    localparam [2:0] INTEGRATE = 3'b000,
                    SPIKE = 3'b010,
                    RESET = 3'b011,
                    REFRACTORY = 3'b100;

    reg [2:0] state;
    
    assign spike_out = spike_out_signal;
    
    wire [7:0] new_potential = (Current_Potential - (Current_Potential >> alpha)) + (spike_in_1 * spike_1_weight) + (spike_in_2 * spike_2_weight) + (spike_in_3 * spike_3_weight);

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            Current_Potential <= 8'd0;
            state <= INTEGRATE;
            spike_out_signal <= 1'b0;
            refactory_counter <= 8'd0;
        end else begin
            case(state)
                INTEGRATE: begin
                    Current_Potential <= new_potential;
                    if (new_potential >= THRESHOLD) begin
                        state <= SPIKE;
                    end else begin
                        spike_out_signal <= 1'b0;
                        state <= INTEGRATE;
                    end
                end

                SPIKE: begin
                    spike_out_signal <= 1'b1;
                    state <= RESET;
                end

                RESET: begin
                    spike_out_signal <= 1'b0;
                    Current_Potential <= 8'd0;
                    state <= REFRACTORY;
                end

                REFRACTORY: begin
                    if (refactory_counter < REFRACTORY_PERIOD) begin
                        refactory_counter <= refactory_counter + 1;
                        state <= REFRACTORY;
                    end else begin
                        refactory_counter <= 8'd0;
                        state <= INTEGRATE;
                    end
                end
            endcase
        end
    end
endmodule
