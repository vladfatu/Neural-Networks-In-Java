package com.ml.nn.layers;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by vladfatu on 27/10/2015.
 */
public class SigmoidLayer extends LinearLayer{

    private double sigmoid(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    private SimpleMatrix sigmoid(SimpleMatrix vector) {
        for (int i=0; i< vector.getNumElements(); i++) {
            vector.set(i, sigmoid(vector.get(0, i)));
        }
        return vector;
    }

    @Override
    public SimpleMatrix forwardPropagate(SimpleMatrix inputVector) {
        SimpleMatrix outputVector = super.forwardPropagate(inputVector);
        outputVector = sigmoid(outputVector);
        return outputVector;
    }
}
