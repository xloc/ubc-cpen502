package cc.xloc.ubc502.activation;

public class TanhActivation implements Activation {
    @Override
    public double f(double x) {
        return Math.tanh(x);
    }

    @Override
    public double dfdx(double x) {
        return 1 - Math.pow(Math.tanh(x), 2);
    }
}
