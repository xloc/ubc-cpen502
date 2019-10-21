package cc.xloc.ubc502;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MatrixMathTest {

    @Test
    void matmul() {
        double[][] A = {
                {1,1,1},
                {2,2,2}
        };

        double[][] B = {
                {2,3},
                {2,3},
                {2,3}
        };

        double[][] C_true = {
                {6,9},
                {12,18}
        };

        double[][] C = MatrixMath.matmul(A, B);
        AssertMatrixEqual(C, C_true);

    }

    @Test
    void matvecmul() {
        double[][] A = {
                {6,9},
                {12,18}
        };

        double[] b = {2,2};

        double[] c_true = {30, 60};

        double[] c = MatrixMath.matvecmul(A, b);
        assertArrayEquals(c, c_true);
    }

    static void AssertMatrixEqual(double[][] A, double[][] B){
        assertEquals(A.length, B.length);
        assertEquals(A[0].length, B[0].length);
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                assertEquals(A[i][j], B[i][j]);
            }
        }
    }
}