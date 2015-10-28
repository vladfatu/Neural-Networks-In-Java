package com.ml.nn;

import com.ml.nn.layers.LinearLayer;
import com.ml.nn.layers.SoftmaxLayer;
import com.ml.nn.networks.FeedForwardNetwork;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

/**
 * Created by vladfatu on 18/09/2015.
 */
public class NeuralNetworkTester {

    public static void main(String[] args) {
//        FeedForwardNetwork network = new FeedForwardNetwork();
//        network.train();

        double x[][] = {
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {5, 9, 8, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {5, 9, 8, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {5, 9, 8, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {5, 9, 8, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {5, 9, 8, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
                {5, 9, 8, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 4, 3, 2, 4, 5, 6, 7},
                {3, 2, 3, 3, 4, 5, 5, 6,3 ,2 ,23, 3, 4, 34, 5, 5, 3, 3,  5, 5, 34, 4, 3, 2, 4, 5, 6, 7},
        };

        double y[][] = {
                {4},
                {9},
                {8},
                {4},
                {4},
                {9},
                {8},
                {4},
                {4},
                {9},
                {8},
                {4},
                {4},
                {9},
                {8},
                {4},
                {4},
                {9},
                {8},
                {4},
                {4},
                {9},
                {8},
                {4},
                {4},
                {9},
                {8},
                {4},

        };

//        Matrix matrix = new Matrix();
//        long timestamp = System.currentTimeMillis();
//
//        for (int i=0; i<1000000; i++) {
//            int z[][] = matrix.multiply(x, y);
//        }
//
//        System.out.println("time for normal: " + (System.currentTimeMillis() - timestamp));
//        timestamp = System.currentTimeMillis();
//
//        SimpleMatrix matrix1 = new SimpleMatrix(x);
//        SimpleMatrix matrix2 = new SimpleMatrix(y);
//
//        for (int i=0; i<1000000; i++) {
//            matrix1.mult(matrix2);
//        }
//
//        System.out.println("time for ejml: " + (System.currentTimeMillis() - timestamp));

//        double x1 = matrix1.get(0, 1);
//        System.out.println(x1);


        SimpleMatrix biasVector = new SimpleMatrix(3,1);
        biasVector.zero();

        SimpleMatrix weightMatrix = new SimpleMatrix(3, 4);
        weightMatrix.zero();
        weightMatrix = weightMatrix.plus(1);
        weightMatrix.print();
        LinearLayer layer = new SoftmaxLayer();
        layer.setWeightMatrix(weightMatrix);
        layer.setBiasVector(biasVector);

        SimpleMatrix inputVector = SimpleMatrix.random(4, 1, 0, 255, new Random());
        inputVector.print();
        layer.forwardPropagate(inputVector).print();

    }


}

class Matrix {
    public int[][] multiply(double[][] m1, double[][] m2) {
        int m1rows = m1.length;
        int m1cols = m1[0].length;
        int m2rows = m2.length;
        int m2cols = m2[0].length;
        if (m1cols != m2rows) {
            throw new IllegalArgumentException("matrices don't match: " + m1cols + " != " + m2rows);
        } else {
            int[][] result = new int[m1rows][m2cols];
            for (int i = 0; i < m1rows; i++) {
                for (int j = 0; j < m2cols; j++) {
                    for (int k = 0; k < m1cols; k++) {
                        result[i][j] += m1[i][k] * m2[k][j];
                    }
                }
            }
            return result;
        }
    }

    /**
     * Matrix print.
     */
    public void mprint(int[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        System.out.println("array[" + rows + "][" + cols + "] = {");
        for (int i = 0; i < rows; i++) {
            System.out.print("{");
            for (int j = 0; j < cols; j++) {
                System.out.print(" " + a[i][j] + ",");
            }
            System.out.println("},");
        }
        System.out.println(":;");
    }
}
