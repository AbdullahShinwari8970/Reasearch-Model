package org.example;

public class Experiments {
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
        // trainAndTestWithHyperparameters();
        trainAndTestSinFunctionWithHyperparameters();
    }

    // New function to train and test for multiple hyperparameter combinations (learning rate, hidden units)
    public static void trainAndTestSinFunctionWithHyperparameters() {
        double[] learningRates = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}; // Different learning rates
        int[] hiddenUnits = {10, 20, 30, 40, 50, 60}; // Different hidden unit sizes
        int maxEpochs = 1000;

        double[][] errors = new double[hiddenUnits.length][learningRates.length];

        // Experiment with different learning rates and hidden units
        for (int i = 0; i < hiddenUnits.length; i++) {
            for (int j = 0; j < learningRates.length; j++) {
                MLP mlp = new MLP(4, hiddenUnits[i], 1); // 4 inputs for the sin function
                double error = trainAndTestSinFunction(mlp, learningRates[j], hiddenUnits[i], maxEpochs);
                errors[i][j] = error; // Store error in matrix
            }
        }

        // Save the results to CSV, passing the learningRates and hiddenUnits arrays
        Utility.saveErrorsToFile(errors, "sin_function_hyperparameter_errors.csv", learningRates, hiddenUnits);
    }

    // Train and test the sin function and return the error
    public static double trainAndTestSinFunction(MLP mlp, double learningRate, int hiddenUnits, int maxEpochs) {
        int numExamples = 500;
        int testExamples = 100;
        double[][] inputs = new double[numExamples][4];
        double[][] targets = new double[numExamples][1];

        // Generating random input vectors and corresponding targets (sin(x1 - x2 + x3 - x4))
        for (int i = 0; i < numExamples; i++) {
            for (int j = 0; j < 4; j++) {
                inputs[i][j] = 2 * Math.random() - 1; // Randomized between -1 and 1
            }
            targets[i][0] = Math.sin(inputs[i][0] - inputs[i][1] + inputs[i][2] - inputs[i][3]);
        }

        // Split into training and testing sets
        double[][] trainInputs = new double[numExamples - testExamples][4];
        double[][] trainTargets = new double[numExamples - testExamples][1];
        double[][] testInputs = new double[testExamples][4];
        double[][] testTargets = new double[testExamples][1];

        System.arraycopy(inputs, 0, trainInputs, 0, numExamples - testExamples);
        System.arraycopy(targets, 0, trainTargets, 0, numExamples - testExamples);
        System.arraycopy(inputs, numExamples - testExamples, testInputs, 0, testExamples);
        System.arraycopy(targets, numExamples - testExamples, testTargets, 0, testExamples);

        // Training loop
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            double totalError = 0;
            for (int i = 0; i < trainInputs.length; i++) {
                mlp.forward(trainInputs[i]);
                totalError += mlp.backward(trainInputs[i], trainTargets[i]);
            }
            mlp.updateWeights(learningRate);

            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + ", Total Error: " + totalError);
            }
        }

        // Test the trained MLP
        double testError = 0;
        for (int i = 0; i < testInputs.length; i++) {
            double[] output = mlp.forward(testInputs[i]);
            testError += Math.pow(output[0] - testTargets[i][0], 2);
        }
        testError /= testInputs.length;
        System.out.println("Test Error: " + testError);
        return testError;
    }
}
