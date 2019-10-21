package cc.xloc.ubc502;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class LayerTest {

    @Test
    void randomInitializeWeight() {
        int nInputs = 10;
        int nOutputs = 5;

        double[][] w = new double[nOutputs][nInputs];
        for (double[] wi: w) {
            Arrays.fill(wi, 0);
        }

        Layer.randomInitializeWeight(w, 0.1, true);
        assertTrue(assert2DArrayNotEqual(w, 0));
    }

    static boolean assert2DArrayEqual(double[][] arr, double val){
        for (double[] ai : arr) {
            for (double aij: ai) {
                if (aij != val) {
                    return false;
                }
            }
        }
        return true;
    }

    static boolean assert2DArrayNotEqual(double[][] arr, double val){
        for (double[] ai : arr) {
            for (double aij: ai) {
                if (aij == val) {
                    return false;
                }
            }
        }
        return true;
    }

    @Test
    void forwardPropagate() {
        Layer l = new Layer(2,2);
        double[] d = {1,1};
        l.weights = new double[][] {
                {1,1},
                {3,-4}
        };
        double[] result = l.forwardPropagate(new double[]{4,4});
        System.out.println(Arrays.toString(result));
    }

    @Test
    void backPropagate() {
    }
}