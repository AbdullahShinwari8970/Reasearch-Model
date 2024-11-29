package org.example;

import java.util.Random;

public class MLP {
    private int NI, NH, NO;
    private double[][] W1, W2;
    private double[][] dW1, dW2;
    private double[] Z1, H, Z2, O;
    private Random random;

    // Constructor
    public MLP(int NI, int NH, int NO) {
        this.NI = NI;
        this.NH = NH;
        this.NO = NO;

        W1 = new double[NI][NH];W2 = new double[NH][NO];
        dW1 = new double[NI][NH];dW2 = new double[NH][NO];

        Z1 = new double[NH];Z2 = new double[NO];
        H = new double[NH];O = new double[NO];

        random = new Random();
        //randomizeWeights(); Continue From Here; Remember Pure Object Oriented.
    }

}


