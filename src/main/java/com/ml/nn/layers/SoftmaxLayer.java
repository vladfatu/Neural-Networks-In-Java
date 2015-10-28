package com.ml.nn.layers;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by vladfatu on 28/10/2015.
 */
public class SoftmaxLayer extends LinearLayer {

    private SimpleMatrix softmax(SimpleMatrix vector) {
        vector = vector.elementExp();
        double sum = vector.elementSum();
        for (int i=0; i< vector.getNumElements(); i++) {
            vector.set(i, vector.get(i, 0) / sum);
        }
        return vector;
    }

    @Override
    public SimpleMatrix forwardPropagate(SimpleMatrix inputVector) {
        SimpleMatrix outputVector = super.forwardPropagate(inputVector);
        outputVector = softmax(outputVector);
        return outputVector;
    }
}
