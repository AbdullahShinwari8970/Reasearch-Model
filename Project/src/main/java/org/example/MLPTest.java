package org.example;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


public class MLPTest {
    @Test
    void testWeightInitializationDimensions() {
        MLP mlp = new MLP(2, 3, 1);

        Assertions.assertEquals(2, mlp.W1.length); //Number of Rows for W1 = 2
        Assertions.assertEquals(3, mlp.W1[0].length); //Number of Columns for W1 = 3

        Assertions.assertEquals(3, mlp.W2.length); //Number of Row for W2 = 3
        Assertions.assertEquals(1, mlp.W2[0].length); //Number of Columns for W2= 1
    }

    @Test
    void testWeightInitializationSmallValues() {
        MLP mlp = new MLP(2, 3, 1);

        boolean w1WithinRange = true;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double value = mlp.W1[i][j];
                if (value < -0.1 || value > 0.1) {
                    w1WithinRange = false;
                }
            }
        }

        boolean w2WithinRange = true;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 1; j++) {
                double value = mlp.W2[i][j];
                if (value < -0.1 || value > 0.1) {
                    w2WithinRange = false;
                }
            }
        }

        //To Test W1 and W2 values are within the range [-0.1, 0.1]
        assertTrue(w1WithinRange);
        assertTrue(w2WithinRange);
    }

    @Test
    void testWeightChangeInitializationToZero() {
        MLP mlp = new MLP(2, 3, 1);

        boolean dW1AllZero = true;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                if (mlp.dW1[i][j] != 0.0) {
                    dW1AllZero = false;
                }
            }
        }

        boolean dW2AllZero = true;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 1; j++) {
                if (mlp.dW2[i][j] != 0.0) {
                    dW2AllZero = false;
                }
            }
        }

        //To Test dW1 and dW2 values are all zeros
        assertTrue(dW1AllZero, "values in dW1 should be zero.");
        assertTrue(dW2AllZero, "values in dW2 should be zero.");
    }

    @Test
    void testForwardPass() {
        MLP mlp = new MLP(2, 3, 1);

        //case generated using gpt and corrected by hand with me.
        mlp.W1 = new double[][]{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
        mlp.W2 = new double[][]{{0.7}, {0.8}, {0.9}};

        double[] input = {1.0, 0.5};
        double[] output = mlp.forward(input);

        assertEquals(1, output.length);

//        //Step 1 is to Compute Z1 = input * W1
//        double[] Z1 = {
//                1.0 * 0.1 + 0.5 * 0.4,  // 0.3
//                1.0 * 0.2 + 0.5 * 0.5,  // 0.45
//                1.0 * 0.3 + 0.5 * 0.6   // 0.6
//        };
//
//        //Step 2 is to Apply sigmoid to Z1
//        double[] A1 = {
//                1 / (1 + Math.exp(-Z1[0])), // sigmoid(0.3)
//                1 / (1 + Math.exp(-Z1[1])), // sigmoid(0.45)
//                1 / (1 + Math.exp(-Z1[2]))  // sigmoid(0.6)
//        };
//
//        //Step 3 is to Compute Z2 = A1 * W2
//        double Z2 = A1[0] * 0.7 + A1[1] * 0.8 + A1[2] * 0.9;
//
//        // Step 4 is to Apply sigmoid to Z2 for final output
//        double expectedOutput = 1 / (1 + Math.exp(-Z2));
//
//        System.out.println("Z1: " +  Arrays.toString(Z1));
//        System.out.println("A1: " +  Arrays.toString(A1));
//        System.out.println("Z2: " + Z2);
//        System.out.println("Output: " + expectedOutput);

        Assertions.assertTrue(Math.abs(output[0] - 0.8133174355027203) < 1e-5, "Forward pass output is incorrect");
    }

}
