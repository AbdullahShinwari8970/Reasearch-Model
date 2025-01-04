package org.example;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

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
}
