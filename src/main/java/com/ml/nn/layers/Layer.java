package com.ml.nn.layers;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by vladfatu on 23/10/2015.
 */
public interface Layer {

    SimpleMatrix forwardPropagate(SimpleMatrix inpuVector);

    void setWeightMatrix(SimpleMatrix weightMatrix);

    SimpleMatrix getWeightMatrix();

    void setBiasVector(SimpleMatrix biasVector);

    SimpleMatrix getBiasVector();

}
