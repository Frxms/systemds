package org.apache.sysds.test.functions.codegen;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;

import java.util.Arrays;

import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.vectDivWrite;

public class SparseRowPerfTest {

	private final int m;
	private final int n;
	private final double sparsity2;
	private final int warmupRuns;
	private final int repetitions;

	private static final double sparsity1 = 0.9;

	public SparseRowPerfTest() {
		this(250, 10000, 50, 5000, 0.07);
	}

	public SparseRowPerfTest(int rl, int cl, int warmupRuns, int repetitions, double sparsity) {
		m = rl;
		n = cl;
		this.warmupRuns = warmupRuns;
		this.repetitions = repetitions;
		this.sparsity2 = sparsity;
	}

	private void testSparseBinaryDivPerf() {
		long sparseResults[] = new long[repetitions];
		long denseResults[] = new long[repetitions];

		System.out.println("Sparsity: " + sparsity2 + "; rl: " + m + "; cl: " + n);

		for(int i = 0; i < warmupRuns; i++) {
			runNanoSparseTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
			runNanoDenseTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
		}
		for(int i = 0; i < repetitions; i++) {
			sparseResults[i] = runNanoSparseTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
			denseResults[i] = runNanoDenseTest(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
		}
		System.out.println("Nanos calc:");
		System.out.println("Sparse calculation: " + Arrays.stream(sparseResults).average().getAsDouble() / 1000000 + " ms");
		System.out.println("Dense calculation: " + Arrays.stream(denseResults).average().getAsDouble() / 1000000 + " ms");

		for(int i = 0; i < warmupRuns; i++) {

		}

		for(int i = 0; i < repetitions; i++) {

		}
	}

	private long runNanoSparseTest(BinType binType, InputType inputType1, InputType inputType2) {
		switch(binType) {
			case VECT_DIV_SCALAR -> {
				return (inputType1 == InputType.SCALAR) ?
					runNanoSparseDivTest(inputType2, true) : runNanoSparseDivTest(inputType1, false);
			}
			case VECT_DIV -> {
				return runNanoSparseDivTest(inputType1, inputType2);
			}
			default -> {
				return Long.MAX_VALUE;
			}
		}
	}

	private long runNanoDenseTest(BinType binType, InputType inputType1, InputType inputType2) {
		switch(binType) {
			case VECT_DIV_SCALAR -> {
				return (inputType1 == InputType.SCALAR) ?
					runNanoDenseDivTest(inputType2, true) : runNanoDenseDivTest(inputType1, false);
			}
			case VECT_DIV -> {
				return runNanoDenseDivTest(inputType1, inputType2);
			}
			default -> {
				return Long.MAX_VALUE;
			}
		}
	}

	private long runNanoSparseDivTest(InputType inputType1, InputType inputType2) {
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 1264);
		double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 1265);
		long nanos = System.nanoTime();
		for(int i = 0; i < m; i++) {
			vectDivWrite(n, inA.getSparseBlock().values(i), inB.max(),
				inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i));
		}
		System.out.println(System.nanoTime() - nanos);
		return System.nanoTime() - nanos;
	}

	private long runNanoSparseDivTest(InputType inputType, boolean scalarVector) {
		double sparsity = (inputType == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock in = MatrixBlock.randOperations(m, n, sparsity, -5, 5, "uniform", 1264);
		if(scalarVector) {
			long nanos = System.nanoTime();
			for(int j = 0; j < m; j++)
				vectDivWrite(n, in.max(), in.getSparseBlock().values(j),
					in.getSparseBlock().indexes(j), in.getSparseBlock().pos(j), in.getSparseBlock().size(j));
			return System.nanoTime() - nanos;
		} else {
			long nanos = System.nanoTime();
			for(int j = 0; j < m; j++)
				vectDivWrite(n, in.getSparseBlock().values(j), in.max(), in.getSparseBlock().indexes(j), in.getSparseBlock().pos(j), in.getSparseBlock().size(j));
			return System.nanoTime() - nanos;
		}
	}

	private long runNanoDenseDivTest(InputType inputType1, InputType inputType2) {
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 1264);
		double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 1265);
		long nanos = System.nanoTime();
		for(int i = 0; i < m; i++) {
			vectDivWrite(inA.getSparseBlock().values(i), inB.max(),
				inA.getSparseBlock().indexes(i), inA.getSparseBlock().pos(i), inA.getSparseBlock().size(i), n);
		}
		return System.nanoTime() - nanos;
	}

	private long runNanoDenseDivTest(InputType inputType, boolean scalarVector) {
		double sparsity = (inputType == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock in = MatrixBlock.randOperations(m, n, sparsity, -5, 5, "uniform", 1264);
		if(scalarVector) {
			long nanos = System.nanoTime();
			for(int j = 0; j < m; j++)
				vectDivWrite(in.max(), in.getSparseBlock().values(j),
					in.getSparseBlock().indexes(j), in.getSparseBlock().pos(j), in.getSparseBlock().size(j), n);
			return System.nanoTime() - nanos;
		} else {
			long nanos = System.nanoTime();
			for(int j = 0; j < m; j++)
				vectDivWrite(in.getSparseBlock().values(j), in.max(), in.getSparseBlock().indexes(j), in.getSparseBlock().pos(j), in.getSparseBlock().size(j), n);
			return System.nanoTime() - nanos;
		}
	}

//	private long runTimingUtilSparseDivTest()

	public static void main(String[] args) {
		new SparseRowPerfTest().testSparseBinaryDivPerf();
	}
}
