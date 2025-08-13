package org.apache.sysds.performance.primitives;

import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;

import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.*;
import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.cleanupSparseThreadLocalMemory;

public class UnaryPrimitivesTest {

	private static final double sparsity1 = 0.9;

	private final double sparsity2;
	private final int m;
	private final int n;

	private SparseBlockMCSR sparseIn;
	private DenseBlock denseIn;
	private double scalar;


	public UnaryPrimitivesTest(int rl, int cl, double sparsity) {
		m = rl;
		n = cl;
		this.sparsity2 = sparsity;
	}

	public String[] primitiveTester(UnaryType uType, InputType inputType, int warmupRuns, int repetitions) {
		getMatrices(inputType);
		System.out.println("Sparsity: " + sparsity2 + "; rl: " + m + "; cl: " + n);

		setupThreadLocalMemory(1, n);
		setupSparseThreadLocalMemory(1, (int) (n*sparsity2 + 100*sparsity2), -1);

		TimingUtils.time(() -> sparseTest(uType, inputType), warmupRuns);
		TimingUtils.time(() -> denseTest(uType, inputType), warmupRuns);

		double[] sparseResults = TimingUtils.time(() -> sparseTest(uType, inputType), repetitions);
		double[] denseResults = TimingUtils.time(() -> denseTest(uType, inputType), repetitions);

		String sparseTime = TimingUtils.stats(sparseResults).split("\\+-")[0];
		String denseTime = TimingUtils.stats(denseResults).split("\\+-")[0];

		cleanupThreadLocalMemory();
		cleanupSparseThreadLocalMemory();

		System.out.println("Sparse calculation: " + TimingUtils.stats(sparseResults));
		System.out.println("Dense calculation " + TimingUtils.stats(denseResults));
		return new String[] {sparseTime, denseTime};
	}

	public void sparseTest(UnaryType uType, InputType inputType) {
		switch(uType) {
			case VECT_SQRT: runSparseSqrt();
			case VECT_ABS: runSparseAbs();
		}
	}

	public void denseTest(UnaryType uType, InputType inputType) {
		switch(uType) {
			case VECT_SQRT: runDenseSqrt();
			case VECT_ABS: runDenseAbs();
		}
	}

	public void runSparseSqrt() {
		for(int i = 0; i < m; i++) {
			vectSqrtWrite(n, sparseIn.values(i), sparseIn.indexes(i), sparseIn.pos(i), sparseIn.size(i));
		}
	}

	public void runSparseAbs() {
		for(int i = 0; i < m; i++) {
			vectAbsWrite(n, sparseIn.values(i), sparseIn.indexes(i), sparseIn.pos(i), sparseIn.size(i));
		}
	}

	public void runDenseSqrt() {
		for(int i = 0; i < m; i++) {
			vectSqrtWrite(sparseIn.values(i), sparseIn.indexes(i), sparseIn.pos(i), sparseIn.size(i), n);
		}
	}

	public void runDenseAbs() {
		for(int i = 0; i < m; i++) {
			vectAbsWrite(sparseIn.values(i), sparseIn.indexes(i), sparseIn.pos(i), sparseIn.size(i), n);
		}
	}

	public void getMatrices(InputType inputType1) {
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		double[][] A = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityA, 1251);
		double[][] B = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityA, 1251);
		MatrixBlock mA = DataConverter.convertToMatrixBlock(A);
		MatrixBlock mB = DataConverter.convertToMatrixBlock(B);

		if(inputType1 == InputType.SCALAR)
			scalar = mA.max();
		//		scalar = -1;

		if(!mA.isInSparseFormat())
			mA.denseToSparse(true);
		sparseIn = new SparseBlockMCSR(mA.getSparseBlock());

		if(mB.isInSparseFormat())
			mB.sparseToDense();
		denseIn = mB.getDenseBlock();
	}
}
