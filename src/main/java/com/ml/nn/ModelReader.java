package com.ml.nn;

import com.ml.nn.layers.Layer;
import com.ml.nn.layers.SigmoidLayer;
import com.ml.nn.layers.SoftmaxLayer;
import com.ml.nn.model.Model;
import org.ejml.simple.SimpleMatrix;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * Created by vladfatu on 29/10/2015.
 */
public class ModelReader {

    public Model readModel() throws IOException {
        Model model = new Model();
        InputStream inputStream = getClass().getClassLoader().getResourceAsStream("layers.txt");
        Scanner scanner = new Scanner(inputStream);
        List<String> lines = new ArrayList<>();
        while (scanner.hasNext()) {
            lines.add(scanner.nextLine());
        }
        for (int i = 0; i < lines.size(); i++) {
            model.addLayer(readLayer(i, lines.get(i)));
        }
        return model;
    }

    private Layer readLayer(int index, String description) throws IOException {
        Layer layer = getLayerForDescription(description);
        layer.setWeightMatrix(readMatrix("weights" + index + ".txt", description));
        layer.setBiasVector(readMatrix("bias" + index + ".txt", layer.getWeightMatrix().getMatrix().numRows, 1));
        return layer;
    }

    private Layer getLayerForDescription(String description) {
        if (description.startsWith("Sigmoid")) {
            return new SigmoidLayer();
        } else {
            return new SoftmaxLayer();
        }
    }

    private SimpleMatrix readMatrix(String path, String description) throws IOException {
        StringTokenizer tokenizer = new StringTokenizer(description);
        tokenizer.nextToken();
        int rows = Integer.decode(tokenizer.nextToken());
        int columns = Integer.decode(tokenizer.nextToken());
        return readMatrix(path, rows, columns);
    }

    private SimpleMatrix readMatrix(String path, int rows, int columns) throws IOException {
        SimpleMatrix simpleMatrix = new SimpleMatrix(rows, columns);

        InputStream inputStream = getClass().getClassLoader().getResourceAsStream(path);
        Scanner scanner = new Scanner(inputStream);
        scanner.nextLine();
        for (int i=0; i<rows; i++) {
            for (int j=0; j<columns; j++) {
                double nextValue = Double.parseDouble(scanner.next());
                simpleMatrix.set(i, j, nextValue);
            }
        }
        return simpleMatrix;
    }

}
