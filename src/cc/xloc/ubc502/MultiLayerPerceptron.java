package cc.xloc.ubc502;

import cc.xloc.ubc502.activation.Activation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Logger;

@SuppressWarnings("WeakerAccess")
public class MultiLayerPerceptron {
    private static final Logger COST_LOGGER = Logger.getLogger("MultiLayerPerceptron:cost");
    private static final Logger ITER_LOGGER = Logger.getLogger("MultiLayerPerceptron:iter");

    private ArrayList<Layer> layers = new ArrayList<>();
    private Layer outputLayer;

    public MultiLayerPerceptron(Activation activation, int ...nNeurons) {
        assert nNeurons.length > 1;
        for (int i = 0; i < nNeurons.length - 1; i++) {
            layers.add(new Layer(activation, nNeurons[i], nNeurons[i+1]));
        }
        outputLayer = layers.get(layers.size()-1);
    }

    public void randomInitializeWeights(double epsilon){
        for (Layer l :layers) {
            l.initializeWeight(epsilon);
        }
    }

    @SuppressWarnings("ForLoopReplaceableByForEach")
    public void train(double[][] X, double[][] y, int maxEpochs, double stopCost, double lr, double momentum) {
        for (int i_epoch = 0; i_epoch < maxEpochs; i_epoch++) {
            double cost = 0;
            double[][] outputs = new double[X.length][];
            for (Layer l :layers) l.startEpoch();
            for (int i = 0; i < X.length; i++) {
                double[] connection = X[i];
                for (int i_layer = 0; i_layer < layers.size(); i_layer++)
                    connection = layers.get(i_layer).forwardPropagate(connection);

                outputs[i] = connection;
                double[] residual = new double[outputLayer.nOutput];
                for (int j = 0; j < outputLayer.nOutput; j++) {
                    residual[j] = (outputs[i][j] - y[i][j]);
                    cost += Math.pow(outputs[i][j] - y[i][j], 2);
                }

                connection = residual;
                for (int i_layer = layers.size()-1; i_layer >= 0; i_layer--)
                    connection = layers.get(i_layer).backPropagate(connection);
            }
            for (Layer l :layers) l.endEpoch(lr, momentum);

//            System.out.println(String.format("Cost:      %.10f", cost));
            COST_LOGGER.info(String.format("%f", cost));
            if (cost <= stopCost) {
                System.out.println(String.format("Cost:      %.10f", cost));
                System.out.println(String.format("Iteration: %d", i_epoch));
                ITER_LOGGER.info(String.format("%d", i_epoch));
                break;
            }

        }
    }

    public void train_2_forwards(double[][] X, double[][] Y, int maxEpochs, double stopCost, double lr, double momentum) {
        Layer l1 = layers.get(0);
        Layer l2 = layers.get(1);

        for (int i_epoch = 0; i_epoch < maxEpochs; i_epoch++) {
            double cost = 0;

            l1.startEpoch();
            l2.startEpoch();

            double[] a1 = null, a2 = null, residual = null;
            // for each element
            for (int i = 0; i < X.length; i++) {
                double[] x = X[i];
                double[] y = Y[i];

                a1 = l1.forwardPropagate(x);
                a2 = l2.forwardPropagate(a1);

                residual = MatrixMath.vecsub(a2, y);

                l2.backPropagate(residual);
            }
            l2.endEpoch(lr, momentum);

            l1.startEpoch();
            // for each element
            for (int i = 0; i < X.length; i++) {
                double[] x = X[i];
                double[] y = Y[i];

                a1 = l1.forwardPropagate(x);
                a2 = l2.forwardPropagate(a1);
                residual = MatrixMath.vecsub(a2, y);
                cost += Arrays.stream(residual).map(a -> a * a).sum();

                double[] residual2 = l2.backPropagate(residual);
                double[] residual3 = l1.backPropagate(residual2);
                System.out.println(Arrays.toString(residual3));
            }
            l1.endEpoch(lr, momentum);

            COST_LOGGER.info(String.format("%f", cost));
            if (cost <= stopCost) {
                System.out.println(String.format("Cost:      %.10f", cost));
                System.out.println(String.format("Iteration: %d", i_epoch));
                break;
            }
        }

    }

    public double[][] predict(double[][] X) {
        double[][] output = X;
        for (Layer l : layers)
            output = l.predict(output);
        return Arrays.stream(output)
                .map(example -> Arrays.stream(example).map(r -> r > 0.5 ? 1 : 0).toArray())
                .toArray(double[][]::new);
    }

    public double evaluate(double[][] X, double[][] y) {
        double[][] y_pred = predict(X);
        int correctCount = 0;
        for (int i = 0; i < y_pred.length; i++) {
            for (int j = 0; j < y_pred[0].length; j++) {
                correctCount += y[i][j] == y_pred[i][j] ? 1 : 0;
            }
        }
        return correctCount * 1.0 / (y.length * y[0].length);
    }
}
