package cc.xloc.ubc502;

import cc.xloc.ubc502.activation.*;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

class MultiLayerPerceptronTest {
    static {
        System.setProperty("java.util.logging.SimpleFormatter.format", "%5$s%n");
    }

    private static int MAX_EPOCHS = 100000;

    @Test
    void trainBinary(){
        useCostLogger(true, "/Users/oliver/Downloads/mlp_binary_sigmoid");
        useIterLogger(false, "/Users/oliver/Downloads/mlp_iters_stats");
        useGeneralLogger(true);

        trainBinary(new SigmoidActivation(), 0.5, 0.2, 0);
    }

    @Test
    void trainBinaryMomentum(){
        useCostLogger(true, "/Users/oliver/Downloads/mlp_binary_momentum");
        useIterLogger(false, "/Users/oliver/Downloads/mlp_iters_stats");
        useGeneralLogger(true);

        trainBinary(new SigmoidActivation(), 0.5, 0.2, 0.9);
    }

    @Test
    void trainBipolarMomentum(){
        useCostLogger(true, "/Users/oliver/Downloads/mlp_bipolar_momentum");
        useIterLogger(false, "/Users/oliver/Downloads/mlp_iters_stats");
        useGeneralLogger(true);

        trainBipolar(new BipolarSigmoidActivation(), 0.5, 0.2, 0.9);
    }

    @Test
    void trainBipolar(){
        useCostLogger(true, "/Users/oliver/Downloads/mlp_bipolar_sigmoid");
        useIterLogger(false, "/Users/oliver/Downloads/mlp_iters_stats");
        useGeneralLogger(true);

        trainBipolar(new BipolarSigmoidActivation(), 0.5, 0.2, 0);
    }

    @Test
    void BinaryActivations(){
        useIterLogger(false, "/Users/oliver/Downloads/mlp_iters_stats");
        useGeneralLogger(true);

        useCostLogger(true, "/Users/oliver/Downloads/mlp_act_sigmoid");
        trainBinary(new SigmoidActivation(), 0.5, 0.2, 0);

        useCostLogger(true, "/Users/oliver/Downloads/mlp_act_smoothedReLU");
        trainBinary(new SmoothedReLUActivation(), 0.5, 0.2, 0);

        MAX_EPOCHS = 1000;
        useCostLogger(true, "/Users/oliver/Downloads/mlp_act_ReLU");
        trainBinary(new ReLUActivation(), 0.5, 0.2, 0);
    }

    @Test
    void BipolarActivations(){
        useIterLogger(false, "/Users/oliver/Downloads/mlp_iters_stats");
        useGeneralLogger(true);

        useCostLogger(true, "/Users/oliver/Downloads/mlp_act_bipolarSigmoid");
        trainBinary(new BipolarSigmoidActivation(), 0.5, 0.2, 0);

        useCostLogger(true, "/Users/oliver/Downloads/mlp_act_tanh");
        trainBinary(new TanhActivation(), 0.5, 0.2, 0);
    }

    @Test
    void learningRates(){
        useIterLogger(false, "/Users/oliver/Downloads/mlp_iters_stats");
        useGeneralLogger(true);

        double[] lrs = {0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50};

        for (double lr : lrs) {
            useCostLogger(true, String.format("/Users/oliver/Downloads/mlp_lr_%.2f", lr));
            trainBinary(new SigmoidActivation(), 0.5, lr, 0);
        }
    }

    @Test
    void momentum(){
        useIterLogger(false, "/Users/oliver/Downloads/mlp_iters_stats");
        useGeneralLogger(true);

        double[] ms = {0, 0.01,0.02,0.05,0.1,0.2,0.5,0.9};

        for (double m : ms) {
            useCostLogger(true, String.format("/Users/oliver/Downloads/mlp_momentum_%.2f", m));
            trainBinary(new SigmoidActivation(), 0.5, 0.2, m);
        }
    }

    @Test
    void trainBinary(Activation activation, double weightInitEpsilon, double lr, double momentum) {
        double[][] X_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] y_train = {{0}, {1}, {1}, {0}};

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(activation, 2, 4, 1);
        mlp.randomInitializeWeights(weightInitEpsilon);
        mlp.train(X_train, y_train, MAX_EPOCHS, 0.05, lr, momentum);
        LOGGER.info(String.format("Accuracy:  %f", mlp.evaluate(X_train, y_train)));
    }

    @Test
    void trainBipolar(Activation activation, double weightInitEpsilon, double lr, double momentum) {
        double[][] X_train = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        double[][] y_train = {{-1}, {1}, {1}, {-1}};

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(activation, 2, 4, 1);
        mlp.randomInitializeWeights(weightInitEpsilon);
        mlp.train(X_train, y_train, MAX_EPOCHS, 0.05, lr, momentum);
        LOGGER.info(String.format("Accuracy:  %f", mlp.evaluate(X_train, y_train)));
    }

    @Test
    void trainBinary_2(Activation activation, double weightInitEpsilon, double lr, double momentum) {
        double[][] X_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] y_train = {{0}, {1}, {1}, {0}};

        MultiLayerPerceptron mlp = new MultiLayerPerceptron(activation, 2, 4, 1);
        mlp.randomInitializeWeights(weightInitEpsilon);
        mlp.train_2_forwards(X_train, y_train, MAX_EPOCHS, 0.05, lr, momentum);
        LOGGER.info(String.format("Accuracy:  %f", mlp.evaluate(X_train, y_train)));
    }

    static void useCostLogger(boolean isUsed, String path) {
        Logger logger = Logger.getLogger("MultiLayerPerceptron:cost");
        logger.setUseParentHandlers(false);
        Arrays.stream(logger.getHandlers()).forEach(logger::removeHandler);

        if (isUsed) {
            try {
                FileHandler fh = new FileHandler(path);
                logger.addHandler(fh);
                SimpleFormatter formatter = new SimpleFormatter();
                fh.setFormatter(formatter);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    static void useIterLogger(boolean isUsed, String path) {
        Logger logger = Logger.getLogger("MultiLayerPerceptron:iter");
        logger.setUseParentHandlers(false);

        if (isUsed) {
            try {
                FileHandler fh = new FileHandler(path);
                logger.addHandler(fh);
                SimpleFormatter formatter = new SimpleFormatter();
                fh.setFormatter(formatter);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private static final Logger LOGGER = Logger.getLogger("MultiLayerPerceptron");
    static void useGeneralLogger(boolean isUsed) {
        if (!isUsed) {
            LOGGER.setUseParentHandlers(false);
        }
    }

}