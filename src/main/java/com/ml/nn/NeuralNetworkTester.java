package com.ml.nn;

import com.ml.nn.networks.FeedForwardNetwork;

/**
 * Created by vladfatu on 18/09/2015.
 */
public class NeuralNetworkTester {

    public static void main(String[] args) {
        FeedForwardNetwork network = new FeedForwardNetwork();
        network.train();

    }

}
