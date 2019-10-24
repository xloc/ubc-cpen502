package cc.xloc.ubc502.activation;

public interface SigmoidLikeActivation extends Activation {
    double dfdx_y(double y);
}
