package cc.xloc.ubc502;

@SuppressWarnings("WeakerAccess")
public class MatrixMath {
    public static double[][] matmul(double[][] A, double[][] B) {
        // A: m x n
        // B: n x d
        int m = A.length, n = A[0].length, d = B[0].length;
        assert n == B.length;

        double[][] result = new double[m][d];
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < d; j++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    public static double[] matvecmul(double[][] A, double[] b) {
        // A: m x n
        // b: n x 1
        int m = A.length, n = A[0].length;
        assert n == b.length;

        double[] result = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i] += A[i][j] * b[j];
            }
        }
        return result;
    }

    public static double[] vecsub(double[] a, double[] b) {
        int len = a.length;
        assert len == b.length;

        double[] result = new double[len];
        for (int i = 0; i < len; i++) {
            result[i] = a[i] - b[i];
        }

        return result;
    }

}
