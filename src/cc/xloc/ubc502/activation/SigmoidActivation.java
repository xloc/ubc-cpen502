package cc.xloc.ubc502.activation;

public class SigmoidActivation implements SigmoidLikeActivation {

    @Override
    public double f(double x) {
        return 1.0 / (1 + Math.exp(-x));
    }

    @Override
    public double dfdx(double x) {
        double y = 1.0 / (1 + Math.exp(-x));
        return y * (1-y);
    }

    @Override
    public double dfdx_y(double y) {
        return y * (1-y);
    }
}
