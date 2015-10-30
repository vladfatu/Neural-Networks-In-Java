package com.ml.nn.analyzers;

import org.ejml.simple.SimpleMatrix;

/**
 * Created by vladfatu on 30/10/2015.
 */
public class MNISTAnalyzer {

    public int getDigit(SimpleMatrix outputVector) {
        double maxValue = 0;
        int maxIndex = -1;
        for (int i=0; i< outputVector.getNumElements(); i++) {
            if (outputVector.get(i, 0) > maxValue) {
                maxIndex = i;
                maxValue = outputVector.get(i, 0);
            }
        }
        return maxIndex;
    }

}
