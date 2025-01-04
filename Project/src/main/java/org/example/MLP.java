package org.example;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MLP {
    private int NI, NH, NO;
    public double[][] W1, W2;
    public double[][] dW1, dW2;
    private double[] Z1, H, Z2, O;
    private Random random;

    private double[] B1; //Biases for hidden layer
    private double[] B2; //Biases for output layer

    //Constructor
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

    //Forward pass ->  Compute outputs for a given input vector
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

    //Backward pass -> Compute weight updates and return error
    public double backward(double[] inputs, double[] targets) {
        double[] outputError = new double[NO];
        double[] hiddenError = new double[NH];

        //Computing output layer error
        for (int k = 0; k < NO; k++) {
            outputError[k] = (O[k] - targets[k]) * O[k] * (1 - O[k]); // Derivative of sigmoid
        }

        //Computing weight updates for W2
        for (int j = 0; j < NH; j++) {
            for (int k = 0; k < NO; k++) {
                dW2[j][k] += H[j] * outputError[k];
            }
        }

        //Computing hidden layer error
        for (int j = 0; j < NH; j++) {
            hiddenError[j] = 0;
            for (int k = 0; k < NO; k++) {
                hiddenError[j] += outputError[k] * W2[j][k];
            }
            hiddenError[j] *= (1 - H[j] * H[j]); // Derivative of tanh
        }

        //Computing weight updates for W1
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NH; j++) {
                dW1[i][j] += inputs[i] * hiddenError[j];
            }
        }

        //Computing total error
        double totalError = 0;
        for (int k = 0; k < NO; k++) {
            totalError += Math.pow(O[k] - targets[k], 2);
        }
        return totalError;

        /*
        I Recived some help online to get this finally working properly.
         */
    }

    //Update weights using accumulated weight changes
    public void updateWeights(double learningRate) {
        for (int i = 0; i < NI; i++) {
            for (int j = 0; j < NH; j++) {
                W1[i][j] -= learningRate * dW1[i][j];
                dW1[i][j] = 0; //Resetting weght change here
            }
        }
        for (int j = 0; j < NH; j++) {
            for (int k = 0; k < NO; k++) {
                W2[j][k] -= learningRate * dW2[j][k];
                dW2[j][k] = 0;
            }
        }
    }


    public static void trainAndTestSinFunction(MLP mlp) {
        int numExamples = 500;
        int testExamples = 100;
        double[][] inputs = new double[numExamples][4];
        double[][] targets = new double[numExamples][1];

        //Generating random input vectors and corresponding targets (sin(x1 - x2 + x3 - x4))
        for (int i = 0; i < numExamples; i++) {
            for (int j = 0; j < 4; j++) {
                inputs[i][j] = 2 * Math.random() - 1; //Randomized between -1 and 1
            }
            targets[i][0] = Math.sin(inputs[i][0] - inputs[i][1] + inputs[i][2] - inputs[i][3]);
        }

        //Split into training and testing sets
        double[][] trainInputs = new double[numExamples - testExamples][4];
        double[][] trainTargets = new double[numExamples - testExamples][1];
        double[][] testInputs = new double[testExamples][4];
        double[][] testTargets = new double[testExamples][1];

        System.arraycopy(inputs, 0, trainInputs, 0, numExamples - testExamples);
        System.arraycopy(targets, 0, trainTargets, 0, numExamples - testExamples);
        System.arraycopy(inputs, numExamples - testExamples, testInputs, 0, testExamples);
        System.arraycopy(targets, numExamples - testExamples, testTargets, 0, testExamples);

        //Training loop
        List<Double> errors = new ArrayList<>();
        double learningRate = 0.01;
        int maxEpochs = 100000;

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < trainInputs.length; i++) {
                mlp.forward(trainInputs[i]);
                totalError += mlp.backward(trainInputs[i], trainTargets[i]);
            }
            mlp.updateWeights(learningRate);
            errors.add(totalError);

            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + ", Total Error: " + totalError);
            }
        }

        Utility.saveErrorsToFile(errors, "errors_sin.csv");

        System.out.println("\nTesting Sin Function:");
        double testError = 0;
        for (int i = 0; i < testInputs.length; i++) {
            double[] output = mlp.forward(testInputs[i]);
            testError += Math.pow(output[0] - testTargets[i][0], 2);
            System.out.printf("Input: [%f, %f, %f, %f], Predicted Output: %.3f, Target: %.3f%n", testInputs[i][0], testInputs[i][1], testInputs[i][2], testInputs[i][3], output[0], testTargets[i][0]);
        }
        testError /= testInputs.length;
        System.out.println("Test Error: " + testError);
    }

    // New function to train and test for multiple hyperparameter combinations (learning rate, hidden units)
    public static void trainAndTestWithHyperparameters() {
        double[] learningRates = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}; // Different learning rates
        int[] hiddenUnits = {10, 20, 30, 40, 50, 60}; // Different hidden unit sizes
        int maxEpochs  = 20;

        double[][] errors = new double[hiddenUnits.length][learningRates.length];

        // Experiment with different learning rates and hidden units
        for (int i = 0; i < hiddenUnits.length; i++) {
            for (int j = 0; j < learningRates.length; j++) {
                MLP mlp = new MLP(2, hiddenUnits[i], 1);
                double error = trainAndTestXOR(mlp, learningRates[j], hiddenUnits[i], maxEpochs);
                errors[i][j] = error; // Store error in matrix
            }
        }

        // Save the results to CSV, passing the learningRates and hiddenUnits arrays
        Utility.saveErrorsToFile(errors, "XOR-hyperparameter_errors.csv", learningRates, hiddenUnits);
    }


    // Train XOR and return the error
    public static double trainAndTestXOR(MLP mlp, double learningRate, int hiddenUnits, int maxEpochs) {
        double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] targets = {{0}, {1}, {1}, {0}};

        double totalError = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            totalError = 0;
            for (int i = 0; i < inputs.length; i++) {
                mlp.forward(inputs[i]);
                totalError += mlp.backward(inputs[i], targets[i]);
            }
            mlp.updateWeights(learningRate);

            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + ", Total Error: " + totalError);
            }
        }

        //Test the trained MLP
        System.out.println("\nTesting:");
        for (int i = 0; i < inputs.length; i++) {
            double[] output = mlp.forward(inputs[i]);
            System.out.printf("Input: [%f, %f], Predicted Output: %.3f%n", inputs[i][0], inputs[i][1], output[0]);
        }
        return totalError;
    }


    public static void main(String[] args) {
        // Run the hyperparameter tuning experiments
        trainAndTestWithHyperparameters();
    }
}


