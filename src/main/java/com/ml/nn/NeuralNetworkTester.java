package com.ml.nn;

import com.ml.nn.model.Model;
import org.ejml.simple.SimpleMatrix;

import java.io.*;
import java.util.Random;

/**
 * Created by vladfatu on 18/09/2015.
 */
public class NeuralNetworkTester {

    public static void main(String[] args) {

//        SimpleMatrix weights = SimpleMatrix.random(4, 784, 0, 255, new Random());
//        try {
//            weights.saveToFileCSV("weights");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        try {
            Model model = ModelReader.readModel();
//            SimpleMatrix inputVector = SimpleMatrix.random(10, 1, 0, 255, new Random());
//            model.validate(inputVector).print();
            MNISTValidator reader = new MNISTValidator();
            reader.validate(model);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

}