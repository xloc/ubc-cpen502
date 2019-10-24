package cc.xloc.ubc502.activation;

public class BipolarSigmoidActivation implements SigmoidLikeActivation {
    @Override
    public double f(double x) {
        return -1 + 2.0 / (1 + Math.exp(-x));
    }

    @Override
    public double dfdx(double x) {
        double y = -1 + 2.0 / (1 + Math.exp(-x));
        return 0.5 * (1 - Math.pow(y, 2));
    }

    @Override
    public double dfdx_y(double y) {
        return 0.5 * (1 - Math.pow(y, 2));
    }
}
