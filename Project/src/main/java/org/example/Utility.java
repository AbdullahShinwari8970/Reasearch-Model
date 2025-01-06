package org.example;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class Utility {
    public static void saveErrorsToFile(double[][] errors, String filename, double[] learningRates, int[] hiddenUnits) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("Hidden Units,Learning Rate,Error\n");
            for (int i = 0; i < errors.length; i++) {
                for (int j = 0; j < errors[i].length; j++) {
                    writer.write(hiddenUnits[i] + "," + learningRates[j] + "," + errors[i][j] + "\n");
                }
            }
            System.out.println("Errors saved to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void saveEpochErrorsToFile(List<Double> epochErrors, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("Epoch,Error\n");
            for (int epoch = 0; epoch < epochErrors.size(); epoch++) {
                writer.write(epoch + "," + epochErrors.get(epoch) + "\n");
            }
            System.out.println("Epoch errors saved to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

