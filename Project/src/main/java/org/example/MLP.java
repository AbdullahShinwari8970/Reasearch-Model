package org.example;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
            H[j] = Math.tanh(Z1[j]); //Activation function is tanh (CHANGE HERE)
        }

        /////////////////////////////

        for (int k = 0; k < NO; k++) { //Going from the Hidden to Output layer
            Z2[k] = 0;
            for (int j = 0; j < NH; j++) {
                Z2[k] += H[j] * W2[j][k];
            }
            O[k] = 1 / (1 + Math.exp(-Z2[k])); //Activation function is sigmoidal (CHANGE HERE)
        }

//        System.out.println("Inputs: " + Arrays.toString(inputs));
//        System.out.println("Weights W1: " + Arrays.deepToString(W1));
//        System.out.println("Z1: " + Arrays.toString(Z1));
//        System.out.println("H (tanh activation): " + Arrays.toString(H));
//        System.out.println("Weights W2: " + Arrays.deepToString(W2));
//        System.out.println("Z2: " + Arrays.toString(Z2));
//        System.out.println("O (sigmoid activation): " + Arrays.toString(O));
//        System.out.println("END OF FORWARD");

        return O;
    }

    //Backward pass -> Compute weight updates and return error
    public double backward(double[] inputs, double[] targets) {
        double[] outputError = new double[NO];
        double[] hiddenError = new double[NH];

        //Computing output layer error
        for (int k = 0; k < NO; k++) {
            outputError[k] = (O[k] - targets[k]) * O[k] * (1 - O[k]); //Derivative of sigmoid
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
            hiddenError[j] *= (1 - H[j] * H[j]); //Derivative of tanh
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
        I Received some help online to get this finally working properly.
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


}


