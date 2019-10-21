package cc.xloc.ubc502;

import org.junit.jupiter.api.Test;

import javax.xml.crypto.dsig.spec.XSLTTransformParameterSpec;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class MultiLayerPerceptronTest {
    @Test
    void train() {
        double[][] X_train = { {0,0}, {0,1}, {1,0}, {1,1} };
        double[][] y_train = { {0},{1},{1},{0} };

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(2,4,1);
        mlp.randomInitializeWeights(1, false);
        mlp.train(X_train, y_train, 100000, 1);
        System.out.println(String.format("Accuracy:  %f", mlp.evaluate(X_train, y_train)));
    }

    @Test
    void train_bipolar() {
//        double[][] X_train = { {-1,-1}, {-1,1}, {1,-1}, {1,1} };
        double[][] X_train = { {0,0}, {0,1}, {1,0}, {1,1} };
        double[][] y_train = { {-1},{1},{1},{-1} };

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(2,4,1);
        mlp.randomInitializeWeights(1, true);
        mlp.train(X_train, y_train, 100000, 1);
        System.out.println(String.format("Accuracy:  %f", mlp.evaluate(X_train, y_train)));
    }
}