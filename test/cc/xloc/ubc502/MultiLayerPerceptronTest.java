package cc.xloc.ubc502;

import cc.xloc.ubc502.activation.Activation;
import cc.xloc.ubc502.activation.BipolarSigmoidActivation;
import cc.xloc.ubc502.activation.SigmoidActivation;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

class MultiLayerPerceptronTest {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "%5$s%n");
    }

    @Test
    void train() {
        useCostLogger(false);
        useIterLogger(true, "updateOnce");
        double[][] X_train = { {0,0}, {0,1}, {1,0}, {1,1} };
        double[][] y_train = { {0},{1},{1},{0} };
        Activation activation = new SigmoidActivation();

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(activation, 2,4,1);
        mlp.randomInitializeWeights(0.5);
        mlp.train(X_train, y_train, 100000, 0.05, 0.2, 0);
        System.out.println(String.format("Accuracy:  %f", mlp.evaluate(X_train, y_train)));
    }

    @Test
    void multipleTrain() {
        for (int i=0; i<100; i++) {
            train();
        }
    }

    @Test
    void multipleTrain_updateTwice() {
        for (int i=0; i<100; i++) {
            train_2_forwards();
        }
    }

    @Test
    void train_bipolar() {
        double[][] X_train = { {-1,-1}, {-1,1}, {1,-1}, {1,1} };
        double[][] y_train = { {-1},{1},{1},{-1} };
        Activation activation = new BipolarSigmoidActivation();

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(activation, 2,4,1);
        mlp.randomInitializeWeights(0.5);
        mlp.train(X_train, y_train, 100000, 0.05, 0.2,0);
        System.out.println(String.format("Accuracy:  %f", mlp.evaluate(X_train, y_train)));
    }

    @Test
    void train_2_forwards() {
        useCostLogger(false);
        useIterLogger(true, "updateTwice");
        double[][] X_train = { {0,0}, {0,1}, {1,0}, {1,1} };
        double[][] y_train = { {0},{1},{1},{0} };
        Activation activation = new SigmoidActivation();

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(activation, 2,4,1);
        mlp.randomInitializeWeights(0.5);
        mlp.train_2_forwards(X_train, y_train, 100000, 0.05, 0.2, 0);
        System.out.println(String.format("Accuracy:  %f", mlp.evaluate(X_train, y_train)));
    }

    static void useCostLogger(boolean isUsed){
        Logger logger = Logger.getLogger("MultiLayerPerceptron:cost");
        logger.setUseParentHandlers(false);

        if (isUsed) {
            try {
                FileHandler fh = new FileHandler("/Users/oliver/Downloads/MLP_costs.txt");
                logger.addHandler(fh);
                SimpleFormatter formatter = new SimpleFormatter();
                fh.setFormatter(formatter);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    static void noLogger() {
        Logger logger = Logger.getLogger("MultiLayerPerceptron:cost");
        logger.setUseParentHandlers(false);
    }

    static void useIterLogger(boolean isUsed, String method){
        Logger logger = Logger.getLogger("MultiLayerPerceptron:iter");
        logger.setUseParentHandlers(false);

        if (isUsed) {
            try {
                FileHandler fh = new FileHandler(
                        String.format("/Users/oliver/Downloads/MLP_iters_%s.txt", method), true);
                logger.addHandler(fh);
                SimpleFormatter formatter = new SimpleFormatter();
                fh.setFormatter(formatter);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}