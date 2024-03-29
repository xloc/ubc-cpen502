package cc.xloc.ubc502;

import cc.xloc.ubc502.activation.Activation;
import cc.xloc.ubc502.activation.SigmoidLikeActivation;

import java.util.Arrays;

@SuppressWarnings("WeakerAccess")
public class Layer {
    public int nInput;
    public int nOutput;
    public Activation activation;

    public double[][] weights;
    public double[] bias;

    public Layer(Activation activation, int nInput, int nOutput) {
        this.nInput = nInput;
        this.nOutput = nOutput;
        this.activation = activation;

        weights = new double[nOutput][nInput+1];
        bias = new double[nInput];
    }

    public void initializeWeight(double epsilon) {
        randomInitializeWeight(weights, epsilon);
    }

    private double[] xAddBias(double[] x){
        double[] x_biased = new double[nInput + 1];
        System.arraycopy(x, 0, x_biased, 0, x.length);
        x_biased[nInput] = 1;
        return x_biased;
    }

    private double[] xRemoveBias(double[] x_biased){
        double[] x = new double[nInput];
        System.arraycopy(x_biased, 0, x, 0, nInput);
        return x;
    }

    public double[] forwardPropagate(double[] x){
        // x: input
        // a: linear-combination W@x
        // y: after-activation f(a)
        this.x = xAddBias(x);
        this.a = MatrixMath.matvecmul(weights, this.x);
        this.y = Arrays.stream(a).map(a_i -> activation.f(a_i)).toArray();
        return y;
    }

    double[][] gradient_w;
    double[][] last_weightChange = null;
    double[] y;
    double[] a;
    double[] x;
    public void startEpoch() {
        if (last_weightChange == null)
            last_weightChange = new double[nOutput][nInput+1];
        gradient_w = new double[nOutput][nInput+1];
        y = null;
        x = null;
    }

    public void endEpoch(double lr, double momentum) {
        for (int i = 0; i < weights.length; i++)
            for (int j = 0; j < weights[0].length; j++) {
                double weightDelta =
                        - lr * gradient_w[i][j]
                        + momentum * last_weightChange[i][j];
                weights[i][j] += weightDelta;
                last_weightChange[i][j] = weightDelta;
            }
    }

    public double[] backPropagate(double[] residual){
        double[] lastLayer_residual = new double[nInput+1];
        for (int k = 0; k < nInput+1; k++) {
            for (int j = 0; j < nOutput; j++) {
                // <?>_p_<?> => d(?) / d(?)
                double cost_p_acti = residual[j];
                double sigmoid_p_wsum = (activation instanceof SigmoidLikeActivation) ?
                        ((SigmoidLikeActivation)activation).dfdx_y(y[j]) : activation.dfdx(a[j]);
                double wsum_p_w = x[k];

                gradient_w[j][k] +=
                        cost_p_acti * sigmoid_p_wsum * wsum_p_w;
                lastLayer_residual[k] +=
                        cost_p_acti * sigmoid_p_wsum * weights[j][k];
            }

        }
        return xRemoveBias(lastLayer_residual);
    }

    public double[][] predict(double[][] X) {
        double[][] results = new double[X.length][];
        for (int i = 0; i < X.length; i++) {
            results[i] = predict(X[i]);
        }
        return results;
    }

    public double[] predict(double[] x){
        double[] result =  MatrixMath.matvecmul(weights, xAddBias(x));
        for (int i = 0; i < result.length; i++) {
            result[i] = activation.f(result[i]);
        }
        return result;
    }

    public static void randomInitializeWeight(double[][] w, double epsilon) {
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                w[i][j] = Math.random() * 2 * epsilon - epsilon;
            }
        }
    }
}
