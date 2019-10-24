package cc.xloc.ubc502.activation;

public class ReLUActivation implements Activation {
    @Override
    public double f(double x) {
        return x > 0 ? x : 0;
    }

    @Override
    public double dfdx(double x) {
        return x > 0 ? x : 0;
    }
}
