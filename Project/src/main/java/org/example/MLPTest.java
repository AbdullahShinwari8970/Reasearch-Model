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

        //Assuming tanh activation for hiddin layer and sigmoidal activation for output layer.
        Assertions.assertTrue(Math.abs(output[0] - 0.7359031528666115) < 1e-5, "Forward pass output is incorrect");
    }


    /*
    Testing Backwards was super hard, given that i make loads of mistakes. Help recieved online.
     */
    @Test
    void testBackwardPass() {
        MLP mlp = new MLP(2, 3, 1);

        mlp.W1 = new double[][]{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
        mlp.W2 = new double[][]{{0.7}, {0.8}, {0.9}};

        double[] input = {1.0, 0.5};
        double[] target = {0.5};

        mlp.forward(input);

        double totalError = mlp.backward(input, target);

        //Expected total error
        double expectedError = Math.pow(0.7359031528666115 - 0.5, 2); // (output - target)^2
        Assertions.assertTrue(Math.abs(totalError - expectedError) < 1e-5, "Total error is incorrect");

        //Check gradients (manually calculated expected values for dW2)
        double[] H = {0.2913126124515909, 0.4218990052500079, 0.5370495669980353}; //Hidden layer outputs
        double[] outputError = {(0.7359031528666115 - 0.5) * 0.7359031528666115 * (1 - 0.7359031528666115)}; //Derivative of sigmoid

        double[][] expectedDW2 = {
                {H[0] * outputError[0]},
                {H[1] * outputError[0]},
                {H[2] * outputError[0]}
        };

        for (int j = 0; j < 3; j++) {//Assertions for dW2
            Assertions.assertTrue(Math.abs(mlp.dW2[j][0] - expectedDW2[j][0]) < 1e-5, "Gradient dW2 mismatch at index " + j);
        }

        //Check gradients (manually calculated expected values for dW1)
        double[] hiddenError = {
                outputError[0] * mlp.W2[0][0] * (1 - H[0] * H[0]), //Derivative off tanh for each hidden unit
                outputError[0] * mlp.W2[1][0] * (1 - H[1] * H[1]),
                outputError[0] * mlp.W2[2][0] * (1 - H[2] * H[2])
        };

        double[][] expectedDW1 = {
                {input[0] * hiddenError[0], input[0] * hiddenError[1], input[0] * hiddenError[2]},
                {input[1] * hiddenError[0], input[1] * hiddenError[1], input[1] * hiddenError[2]}
        };


        for (int i = 0; i < 2; i++) { //Assertions for dW1
            for (int j = 0; j < 3; j++) {
                Assertions.assertTrue(Math.abs(mlp.dW1[i][j] - expectedDW1[i][j]) < 1e-5, "Gradient dW1 mismatch at index [" + i + "][" + j + "]");
            }
        }
    }


}
