package com.ml.nn;

import com.ml.nn.layers.Layer;
import com.ml.nn.layers.SigmoidLayer;
import com.ml.nn.layers.SoftmaxLayer;
import com.ml.nn.model.Model;
import org.ejml.simple.SimpleMatrix;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Created by vladfatu on 29/10/2015.
 */
public class ModelReader {

    public static Model readModel() throws IOException {
        Model model = new Model();
        Path path = Paths.get("layers", "layers.txt");
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        for (int i = 0; i < lines.size(); i++) {
            model.addLayer(readLayer(i, lines.get(i)));
        }
        return model;
    }

    private static Layer readLayer(int index, String description) throws IOException {
        Layer layer = getLayerForDescription(description);
        layer.setWeightMatrix(readMatrix("layers/weights" + index + ".txt", description));
        layer.setBiasVector(readMatrix("layers/bias" + index + ".txt", layer.getWeightMatrix().getMatrix().numRows, 1));
        return layer;
    }

    private static Layer getLayerForDescription(String description) {
        if (description.startsWith("Sigmoid")) {
            return new SigmoidLayer();
        } else {
            return new SoftmaxLayer();
        }
    }

    private static SimpleMatrix readMatrix(String path, String description) throws IOException {
        StringTokenizer tokenizer = new StringTokenizer(description);
        tokenizer.nextToken();
        int rows = Integer.decode(tokenizer.nextToken());
        int columns = Integer.decode(tokenizer.nextToken());
        return readMatrix(path, rows, columns);
    }

    private static SimpleMatrix readMatrix(String path, int rows, int columns) throws IOException {
        SimpleMatrix simpleMatrix = new SimpleMatrix(rows, columns);
        simpleMatrix = simpleMatrix.loadCSV(path);
        return simpleMatrix;
    }

}
