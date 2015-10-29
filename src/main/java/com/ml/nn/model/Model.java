package com.ml.nn.model;

import com.ml.nn.layers.Layer;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vladfatu on 29/10/2015.
 */
public class Model {

    private List<Layer> layers;

    public Model() {
        layers = new ArrayList<Layer>();
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public SimpleMatrix validate(SimpleMatrix input) {
        SimpleMatrix nextInput = input;
        for (Layer layer : layers) {
            nextInput = layer.forwardPropagate(nextInput);
        }
        return nextInput;
    }

}
