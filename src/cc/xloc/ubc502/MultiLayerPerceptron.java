package cc.xloc.ubc502;

import java.util.Arrays;

@SuppressWarnings("WeakerAccess")
public class MultiLayerPerceptron {

    private Layer l1;
    private Layer l2;

    public MultiLayerPerceptron(int nInput, int nHidden, int nOutput) {
        l1 = new Layer(nInput, nHidden);
        l2 = new Layer(nHidden, nOutput);
    }

    public void randomInitializeWeights(double epsilon, boolean isBilinear){
        l1.initializeWeight(epsilon, isBilinear);
        l2.initializeWeight(epsilon, isBilinear);
    }

    public void train(double[][] X, double[][] y, int maxEpochs, double lr) {
        for (int i_epoch = 0; i_epoch < maxEpochs; i_epoch++) {
            double cost = 0;
            double[][] outputs = new double[X.length][];
            l1.startEpoch();
            l2.startEpoch();
            for (int i = 0; i < X.length; i++) {
                double[] hidden_output = l1.forwardPropagate(X[i]);
                outputs[i] = l2.forwardPropagate(hidden_output);

                double[] residual = new double[l2.nOutput];
                for (int j = 0; j < l2.nOutput; j++) {
                    residual[j] = (outputs[i][j] - y[i][j]);
                    cost += Math.pow(outputs[i][j] - y[i][j], 2);
                }

                double[] hiddener_residual = l2.backPropagate(residual);
                @SuppressWarnings("unused")
                double[] input_residual = l1.backPropagate(hiddener_residual); // what is that?
            }
            l1.endEpoch(lr);
            l2.endEpoch(lr);

            if (cost <= 1e-2) {
                System.out.println(String.format("Cost:      %.10f", cost));
                System.out.println(String.format("Iteration: %d", i_epoch));
                break;
            }

        }
    }

    public double[][] predict(double[][] X) {
        return Arrays.stream(l2.predict(l1.predict(X)))
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
