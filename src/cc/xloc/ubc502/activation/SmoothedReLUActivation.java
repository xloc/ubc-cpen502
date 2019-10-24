package cc.xloc.ubc502.activation;

public class SmoothedReLUActivation implements Activation {
    private final double a;

    public SmoothedReLUActivation(double a) {
        this.a = a;
    }

    public SmoothedReLUActivation() {
        this.a = 1;
    }

    @Override
    public double f(double x) {
        return Math.log(1 + Math.exp(a*x));
    }

    @Override
    public double dfdx(double x) {
        return a - a / (1 + Math.exp(a*x));
    }
}
