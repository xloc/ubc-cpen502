package cc.xloc.ubc502.activation;

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class ActivationTest {

    @Test
    void sigmoid() {
        testSigmoidLikeActivation(new BipolarSigmoidActivation());
        testSigmoidLikeActivation(new SigmoidActivation());
        testActivation(new TanhActivation());
    }

    void testActivation(Activation act) {
        double[] test_x = {-0.2, 0.1, 0.3, 0.7};
        double e = 1e-6;
        double threshold = 1e-8;
        for (double x : test_x) {
            double numerical_d = (act.f(x+e) - act.f(x-e))/(2*e);
            double symbol_d = act.dfdx(x);

            double error = Math.abs(numerical_d - symbol_d);
            assertTrue(error < threshold);
//            System.out.println(error);
        }
    }

    void testSigmoidLikeActivation(SigmoidLikeActivation act) {
        double[] test_x = {-0.2, 0.1, 0.3, 0.7};
        double e = 1e-6;
        double threshold = 0.01;
        for (double x : test_x) {
            double numerical_d = (act.f(x+e) - act.f(x-e))/(2*e);
            double symbol_d = act.dfdx(x);
            double symbol_d_y = act.dfdx_y(act.f(x));

            double error;
            error = Math.abs(numerical_d - symbol_d);
            assertTrue(error < threshold);
            error = Math.abs(numerical_d - symbol_d_y);
            assertTrue(error < threshold);
//            System.out.println(error);
        }
    }
}