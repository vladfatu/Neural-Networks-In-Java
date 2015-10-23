package com.ml.nn.layers;

import mikera.arrayz.NDArray;

/**
 * Created by vladfatu on 23/10/2015.
 */
public interface Layer {

    NDArray forwardPropagate();

}
