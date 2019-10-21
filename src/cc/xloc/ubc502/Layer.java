package cc.xloc.ubc502;

@SuppressWarnings("WeakerAccess")
public class Layer {
    public int nInput;
    public int nOutput;

    public double[][] weights;

    public Layer(int nInput, int nOutput) {
        this.nInput = nInput;
        this.nOutput = nOutput;

        weights = new double[nOutput][nInput];
    }

    public void initializeWeight(double epsilon, boolean isBilinear) {
        randomInitializeWeight(weights, epsilon, isBilinear);
    }

    public double[] forwardPropagate(double[] x){
        double[] y = MatrixMath.matvecmul(weights, x);
        for (int i = 0; i < y.length; i++) {
            y[i] = activation(y[i]);
        }
        this.y = y;
        this.x = x;
        return y;
    }

    double[][] gradient_w;
    double[] y;
    double[] x;
    public void startEpoch() {
        gradient_w = new double[nOutput][nInput];
        y = null;
        x = null;
    }

    public void endEpoch(double lr) {
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[0].length; j++)
                weights[i][j] -= lr * gradient_w[i][j];
    }

    public double[] backPropagate(double[] residual){
        double[] lastLayer_residual = new double[nInput];
        for (int k = 0; k < nInput; k++) {
            for (int j = 0; j < nOutput; j++) {
                // <?>_p_<?> => d(?) / d(?)
                double cost_p_acti = residual[j];
                double sigmoid_p_wsum = y[j] * (1 - y[j]);
                double wsum_p_w = x[k];

                gradient_w[j][k] +=
                        cost_p_acti * sigmoid_p_wsum * wsum_p_w;
                lastLayer_residual[k] +=
                        cost_p_acti * sigmoid_p_wsum * weights[j][k];
            }

        }
        return lastLayer_residual;
    }

    public static double activation(double a) {
        return 1.0 / (1.0 + Math.exp(-a));
    }

    public double[][] predict(double[][] X) {
        double[][] results = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            results[i] = predict(X[i]);
        }
        return results;
    }

    public double[] predict(double[] x){
        double[] result =  MatrixMath.matvecmul(weights, x);
        for (int i = 0; i < result.length; i++) {
            result[i] = activation(result[i]);
        }
        return result;
    }

    public static void randomInitializeWeight(double[][] w, double epsilon, boolean isBilinear){
        if (!isBilinear)
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    w[i][j] = Math.random() * epsilon;
                }
            }
        else
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    w[i][j] = Math.random() * 2 * epsilon - epsilon;
                }
            }
    }
}
