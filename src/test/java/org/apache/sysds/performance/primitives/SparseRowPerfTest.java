package org.apache.sysds.performance.primitives;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;

import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.vectDivWrite;

public class SparseRowPerfTest {

	private final int m;
	private final int n;
	private final int warmupRuns;
	private final int repetitions;
	private final int testSize;
	private final double maxSparsity;

	public SparseRowPerfTest() {
		this(1000, 10000, 50, 1000, 0.39, 5);
	}

	public SparseRowPerfTest(int rl, int cl, int warmupRuns, int repetitions, double sparsity, int testSize) {
		m = rl;
		n = cl;
		this.warmupRuns = warmupRuns;
		this.repetitions = repetitions;
		this.maxSparsity = sparsity;
		this.testSize = testSize;
	}

	private void testPrimitivePerf() {
		double[] sparsityVals = sparsityValues(false);
		for(int k = 0; k < testSize; k++) {
			PrimitivesTest tester = new PrimitivesTest(m, n, sparsityVals[k]);
			System.out.println("Sparsity: " + sparsityVals[k] + "; rl: " + m + "; cl: " + n);



			TimingUtils.time(() -> tester.sparseTest(BinType.VECT_DIV, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE), warmupRuns);
			TimingUtils.time(() -> tester.denseTest(BinType.VECT_DIV, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE), warmupRuns);

			double[] sparseResults = TimingUtils.time(() -> tester.sparseTest(BinType.VECT_DIV, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE), repetitions);
			double[] denseResults = TimingUtils.time(() -> tester.denseTest(BinType.VECT_DIV, InputType.VECTOR_SPARSE, InputType.VECTOR_DENSE), repetitions);
			System.out.println("Sparse calculation: " + TimingUtils.stats(sparseResults));
			System.out.println("Dense calculation " + TimingUtils.stats(denseResults));
		}
	}

	private double[] sparsityValues(boolean exp) {
		double[] sparsity = new double[testSize];
		double currVal = maxSparsity;
		if(exp) {
			for(int i = 1; i < testSize; i++) {
				sparsity[i-1] = currVal;
				currVal = currVal * Math.exp(-1*i);
			}
			sparsity[testSize-1] = currVal * Math.exp(-1*testSize-1);
		}else {
			for(int i = 0; i < testSize; i++) {
				sparsity[i] = currVal;
				currVal /= 3;
			}
		}
		return sparsity;
	}

	public static void main(String[] args) {
		new SparseRowPerfTest().testPrimitivePerf();
	}
}
