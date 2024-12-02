package org.example;

import java.util.Random;

public class MLP {
    private int NI, NH, NO;
    public double[][] W1, W2;
    public double[][] dW1, dW2;
    private double[] Z1, H, Z2, O;
    private Random random;

    private double[] B1; //Biases for hidden layer
    private double[] B2; //Biases for output layer

    // Constructor
    public MLP(int NI, int NH, int NO) {
        this.NI = NI;
        this.NH = NH;
        this.NO = NO;

        W1 = new double[NI][NH];W2 = new double[NH][NO];
        dW1 = new double[NI][NH];dW2 = new double[NH][NO];

        Z1 = new double[NH];Z2 = new double[NO];
        H = new double[NH];O = new double[NO];

        B1 = new double[NH];
        B2 = new double[NO];

        //All array elements above have been initalized to 0 (JAVA TING)

        random = new Random();
        randomizeWeights(); //So this method only needs to update W1 and W2 to small values.
    }

    //Initializing the weights to small random values in between and including [-0.1, 0.1]
    private void randomizeWeights() {
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NH; j++) {
                W1[i][j] = (random.nextDouble() * 0.2) - 0.1;
            }
        }
        for (int i = 0; i < NH; i++) {
            for (int j = 0; j < NO; j++) {
                W2[i][j] = (random.nextDouble() * 0.2) - 0.1;
            }
        }
    }

    // Forward pass: Compute outputs for a given input vector
    public double[] forward(double[] inputs) {
        for (int j = 0; j < NH; j++) { //Going from the Input to Hidden Layer
            Z1[j] = 0;
            for (int i = 0; i < NI; i++) {
                Z1[j] += inputs[i] * W1[i][j];
            }
            H[j] = Math.tanh(Z1[j]); //Activation function is tanh
        }

        /////////////////////////////

        for (int k = 0; k < NO; k++) { //Going from the Hidden to Output layer
            Z2[k] = 0;
            for (int j = 0; j < NH; j++) {
                Z2[k] += H[j] * W2[j][k];
            }
            O[k] = 1 / (1 + Math.exp(-Z2[k])); //Activation function is sigmoidal
        }
        return O;
    }

    // Backward pass: Compute weight updates and return error
    public double backward(double[] inputs, double[] targets) {
        double[] outputError = new double[NO];
        double[] hiddenError = new double[NH];

        // Compute output layer error
        for (int k = 0; k < NO; k++) {
            outputError[k] = (O[k] - targets[k]) * O[k] * (1 - O[k]); // Derivative of sigmoid
        }

        // Compute weight updates for W2
        for (int j = 0; j < NH; j++) {
            for (int k = 0; k < NO; k++) {
                dW2[j][k] += H[j] * outputError[k];
            }
        }

        // Compute hidden layer error
        for (int j = 0; j < NH; j++) {
            hiddenError[j] = 0;
            for (int k = 0; k < NO; k++) {
                hiddenError[j] += outputError[k] * W2[j][k];
            }
            hiddenError[j] *= (1 - H[j] * H[j]); // Derivative of tanh
        }

        // Compute weight updates for W1
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NH; j++) {
                dW1[i][j] += inputs[i] * hiddenError[j];
            }
        }

        // Compute total error
        double totalError = 0;
        for (int k = 0; k < NO; k++) {
            totalError += Math.pow(O[k] - targets[k], 2);
        }
        return totalError;
    }

    // Update weights using accumulated weight changes
    public void updateWeights(double learningRate) {
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NH; j++) {
                W1[i][j] -= learningRate * dW1[i][j];
                dW1[i][j] = 0; // Reset weight change
            }
        }
        for (int j = 0; j < NH; j++) {
            for (int k = 0; k < NO; k++) {
                W2[j][k] -= learningRate * dW2[j][k];
                dW2[j][k] = 0; // Reset weight change
            }
        }
    }

    // Main method to test the MLP
    public static void main(String[] args) {
        // XOR problem setup
        double[][] inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
        double[][] targets = { {0}, {1}, {1}, {0} };

        // Initialize the MLP with 2 inputs, 3 hidden units, and 1 output
        MLP mlp = new MLP(2, 3, 1);

        double learningRate = 0.1;
        int maxEpochs = 10000;

        // Training loop
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < inputs.length; i++) {
                mlp.forward(inputs[i]);
                totalError += mlp.backward(inputs[i], targets[i]);
            }
            mlp.updateWeights(learningRate);

            // Print error every 1000 epochs
            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + ", Total Error: " + totalError);
            }
        }

        // Test the trained MLP
        System.out.println("\nTesting:");
        for (int i = 0; i < inputs.length; i++) {
            double[] output = mlp.forward(inputs[i]);
            System.out.printf("Input: [%f, %f], Predicted Output: %.3f%n", inputs[i][0], inputs[i][1], output[0]);
        }
    }
}


