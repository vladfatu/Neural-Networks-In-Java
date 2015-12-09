package com.ml.nn;

import com.ml.nn.analyzers.MNISTAnalyzer;
import com.ml.nn.model.Model;
import org.ejml.simple.SimpleMatrix;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class MNISTValidator {

    public void validate(Model model) throws IOException {
        DataInputStream labels = new DataInputStream(new FileInputStream("mnist-test-set/labels"));
        DataInputStream images = new DataInputStream(new FileInputStream("mnist-test-set/images"));
        int magicNumber = labels.readInt();
        if (magicNumber != 2049) {
            System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
            System.exit(0);
        }
        magicNumber = images.readInt();
        if (magicNumber != 2051) {
            System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
            System.exit(0);
        }
        int numLabels = labels.readInt();
        int numImages = images.readInt();
        checkImagesAndLabelsCounts(numLabels, numImages);

        int numRows = images.readInt();
        int numCols = images.readInt();

        long start = System.currentTimeMillis();
        int numImagesRead = 0;
        int validImages = 0;
        while (labels.available() > 0 && numImagesRead < numLabels) {
            byte label = labels.readByte();
            numImagesRead++;
            double[][] image = new double[numCols][numRows];
            for (int colIdx = 0; colIdx < numCols; colIdx++) {
                for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
                    image[colIdx][rowIdx] = images.readUnsignedByte();
                }
            }

            boolean accurate = validateImage(model, image, label);
            if (accurate) {
                validImages++;
            }

        }
        System.out.println();
        long end = System.currentTimeMillis();
        long elapsed = end - start;
        System.out.println("Validated " + numLabels + " samples in " + elapsed + " milliseconds");
        System.out.println("Accuracy: " + (((double)validImages/numLabels) * 100) + "%");
    }

    private boolean validateImage(Model model, double[][] image, byte label) {
        SimpleMatrix inputVector = new SimpleMatrix(image);
//        printMatrix(inputVector);
//        System.out.println();
        inputVector.reshape(784, 1);
        SimpleMatrix outputVector = model.validate(inputVector);
//        outputVector.print();
        MNISTAnalyzer analyzer = new MNISTAnalyzer();
        int digit = analyzer.getDigit(outputVector);
        return digit == label;
    }

    private static void printMatrix(SimpleMatrix mat) {
        for (int i=0; i<28; i++) {
            for (int j=0; j<28; j++) {
                if (mat.get(i, j) > 0) {
                    System.out.print(" ");
                } else {
                    System.out.print("0");
                }
            }
            System.out.println();
        }
    }

    private void checkImagesAndLabelsCounts(int numLabels, int numImages) {
        if (numLabels != numImages) {
            System.err.println("Image file and label file do not contain the same number of entries.");
            System.err.println("  Label file contains: " + numLabels);
            System.err.println("  Image file contains: " + numImages);
            throw new RuntimeException();
        }
    }

}
