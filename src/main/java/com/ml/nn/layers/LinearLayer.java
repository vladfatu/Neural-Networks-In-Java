package com.ml.nn.layers;


import org.ejml.simple.SimpleMatrix;

/**
 * Created by vladfatu on 27/10/2015.
 */
public class LinearLayer implements Layer {

    private SimpleMatrix weightMatrix;
    private SimpleMatrix biasVector;

    public SimpleMatrix forwardPropagate(SimpleMatrix inputVector) {
        return weightMatrix.mult(inputVector).plus(biasVector);
    }

    public SimpleMatrix getWeightMatrix() {
        return weightMatrix;
    }

    public void setWeightMatrix(SimpleMatrix weightMatrix) {
        this.weightMatrix = weightMatrix;
    }

    public SimpleMatrix getBiasVector() {
        return biasVector;
    }

    public void setBiasVector(SimpleMatrix biasVector) {
        this.biasVector = biasVector;
    }

}
