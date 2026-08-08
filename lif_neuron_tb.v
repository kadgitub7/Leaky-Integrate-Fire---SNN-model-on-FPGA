`timescale 1ns / 1ps

module lif_neuron_tb();
    reg clk;
    reg reset;

    reg spike_in_1;
    reg spike_in_2;
    reg spike_in_3;
    reg [7:0] spike_1_weight;
    reg [7:0] spike_2_weight;
    reg [7:0] spike_3_weight;

    reg [7:0] alpha;

    wire spike_out;
    
    lif_neuron uut(
        .clk(clk),
        .reset(reset),
        .spike_in_1(spike_in_1),
        .spike_in_2(spike_in_2),
        .spike_in_3(spike_in_3),
        .spike_1_weight(spike_1_weight),
        .spike_2_weight(spike_2_weight),
        .spike_3_weight(spike_3_weight),
        .alpha(alpha)
    );
    
    always #5 clk = ~clk; // 100MHz clock
    
    initial begin
        $monitor("Spike1 = %b | Spike1Weight = %d || Spike2 = %b | Spike2Weight = %d || Spike3 = %b | Spike3Weight = %d | alpha = %d | CurrentPortential = %d | OutputSignal = %b", spike_in_1, spike_1_weight, spike_in_2, spike_2_weight, spike_in_3, spike_3_weight, alpha, uut.Current_Potential, uut.spike_out);
        
        clk = 0;
        reset = 1;
        alpha = 8'd4;
        spike_in_1 = 1'b0;
        spike_in_2 = 1'b0;
        spike_in_3 = 1'b0;
        spike_1_weight = 8'd10;
        spike_2_weight = 8'd11;
        spike_3_weight = 8'd9;
        #10;
        reset = 0;
        
        spike_in_1 = 1'b1;
        #10;
        
        spike_in_1 = 1'b0;
        spike_in_2 = 1'b1;
        #10;
        
        spike_in_2 = 1'b0;
        spike_in_3 = 1'b1;
        #10;
        
        spike_in_3 = 1'b0;
        spike_in_2 = 1'b1;
        #10;
        
        spike_in_2 = 1'b0;
        spike_in_1 = 1'b1;
        #10;
        
        spike_in_1 = 1'b0;
        #10;
        
        spike_in_1 = 1'b1;
        #10;
        
        spike_in_1 = 1'b0;
        spike_in_2 = 1'b1;
        #10;
        
        spike_in_2 = 1'b0;
        spike_in_3 = 1'b1;
        #10;
        
        spike_in_3 = 1'b0;
        spike_in_2 = 1'b1;
        #10;
        
        spike_in_2 = 1'b0;
        spike_in_1 = 1'b1;
        #10;
        
        spike_in_1 = 1'b0;
        #10;
        
        spike_in_1 = 1'b1;
        #10;
        
        spike_in_1 = 1'b0;
        spike_in_2 = 1'b1;
        #10;
        
        spike_in_2 = 1'b0;
        spike_in_3 = 1'b1;
        #10;
        
        spike_in_3 = 1'b0;
        spike_in_2 = 1'b1;
        #10;
        
        spike_in_2 = 1'b0;
        spike_in_1 = 1'b1;
        #10;
        
        spike_in_1 = 1'b0;
        #10;
        
        spike_in_1 = 1'b1;
        #10;
        
        spike_in_1 = 1'b0;
        spike_in_2 = 1'b1;
        #10;
        
        spike_in_2 = 1'b0;
        spike_in_3 = 1'b1;
        #10;
        
        spike_in_3 = 1'b0;
        spike_in_2 = 1'b1;
        #10;
        
        spike_in_2 = 1'b0;
        spike_in_1 = 1'b1;
        #10;
        
        spike_in_1 = 1'b0;
        #10;
        
    end
endmodule
