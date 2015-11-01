package com.ml.nn;

import com.ml.nn.analyzers.MNISTAnalyzer;
import com.ml.nn.model.Model;
import org.ejml.simple.SimpleMatrix;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;

/**
 * Created by Vlad on 10/31/2015.
 */
public class ImageTester {

    public static void main(String[] args) {
        BufferedImage image = loadImage("test-images" + File.separator + "9.jpg");
        image = getScaledImage(image, 28, 28);
        long timestamp = System.currentTimeMillis();
        double[][] matrixImage = convertTo2DUsingGetRGB(image);
        validateImage(matrixImage);
//        convertTo2DWithoutUsingGetRGB(image);
        System.out.println("without getRGB: " + (System.currentTimeMillis() - timestamp));
//        saveImage(image, "processed.png");

    }

    private static void validateImage(double[][] image) {
        try {
            Model model = ModelReader.readModel();
            SimpleMatrix input = new SimpleMatrix(image);
            input = input.transpose();
//            input.print();
            printMatrix(input);
//            printMatrix(image);
            input.reshape(784, 1);
            SimpleMatrix outputVector = model.validate(input);
            outputVector.print();
            MNISTAnalyzer analyzer = new MNISTAnalyzer();
            int digit = analyzer.getDigit(outputVector);
            System.out.println("Detected digit: " + digit);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void printMatrix(double[][] mat) {
        for (int i=0; i<28; i++) {
            for (int j=0; j<28; j++) {
                if (mat[i][j] > 0) {
                    System.out.print("x");
                } else {
                    System.out.print(" ");
                }
            }
            System.out.println();
        }
    }

    private static void printMatrix(SimpleMatrix mat) {
        for (int i=0; i<28; i++) {
            for (int j=0; j<28; j++) {
                if (mat.get(i, j) > 0) {
                    System.out.print(" ");
                } else {
                    System.out.print("0");
                }
            }
            System.out.println();
        }
    }

    private static BufferedImage loadImage(String path) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(path));
        } catch (IOException e) {
        }
        return img;
    }

    private static void saveImage(BufferedImage image, String path) {
        try {
            File outputFile = new File(path);
            ImageIO.write(image, "png", outputFile);
        } catch (IOException e) {
        }
    }

    private static BufferedImage getScaledImage(BufferedImage initialImage, int width, int height) {
        Image tmp = initialImage.getScaledInstance(width, height, BufferedImage.SCALE_FAST);
        BufferedImage scaledImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        scaledImage.getGraphics().drawImage(tmp, 0, 0, null);
        return scaledImage;
    }

    private static double[][] convertTo2DUsingGetRGB(BufferedImage image) {
        Raster raster = image.getData();
        int width = image.getWidth();
        int height = image.getHeight();
        double[][] result = new double[height][width];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int pixel = 255 - raster.getSample(row, col, 0);
                if (pixel < 110) {
                    pixel = 0;
                }
                result[row][col] = pixel;

            }
        }

        return result;
    }

    private static int[][] convertTo2DWithoutUsingGetRGB(BufferedImage image) {

        final byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        final int width = image.getWidth();
        final int height = image.getHeight();
        final boolean hasAlphaChannel = image.getAlphaRaster() != null;

        int[][] result = new int[height][width];
        if (hasAlphaChannel) {
            final int pixelLength = 4;
            for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel += pixelLength) {
                int argb = 0;
                argb += (((int) pixels[pixel] & 0xff) << 24); // alpha
                argb += ((int) pixels[pixel + 1] & 0xff); // blue
                argb += (((int) pixels[pixel + 2] & 0xff) << 8); // green
                argb += (((int) pixels[pixel + 3] & 0xff) << 16); // red
                result[row][col] = argb;
                col++;
                if (col == width) {
                    col = 0;
                    row++;
                }
            }
        } else {
            final int pixelLength = 3;
            for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel += pixelLength) {
                int argb = 0;
                argb += -16777216; // 255 alpha
                argb += ((int) pixels[pixel] & 0xff); // blue
                argb += (((int) pixels[pixel + 1] & 0xff) << 8); // green
                argb += (((int) pixels[pixel + 2] & 0xff) << 16); // red
                result[row][col] = argb;
                col++;
                if (col == width) {
                    col = 0;
                    row++;
                }
            }
        }

        return result;
    }
}
