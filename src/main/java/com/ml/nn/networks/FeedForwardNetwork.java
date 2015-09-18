package com.ml.nn.networks;

import mikera.arrayz.NDArray;
import mikera.randomz.Randomz;

/**
 * Created by vladfatu on 18/09/2015.
 */
public class FeedForwardNetwork {

    public FeedForwardNetwork() {
    }
    
    public void train() {
        
    }

    public void feedForward() {

    }

    public void backpropagate() {
        
    }

    public void getCost() {

    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    private double sigmoidPrime(double z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }

}
