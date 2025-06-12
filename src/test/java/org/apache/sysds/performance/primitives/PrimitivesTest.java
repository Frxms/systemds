package org.apache.sysds.performance.primitives;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;
import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.vectDivWrite;

public class PrimitivesTest {

	private static final double sparsity1 = 0.9;

	private final double sparsity2;
	private final int m;
	private final int n;

	private SparseBlockMCSR sparseInA;
	private SparseBlockMCSR sparseInB;
	private DenseBlock denseIn;
	private double scalar;


	public PrimitivesTest(int rl, int cl, double sparsity) {
		m = rl;
		n = cl;
		this.sparsity2 = sparsity;
	}

	public String[] primitiveTester(BinType binType, InputType inputType1, InputType inputType2, int warmupRuns, int repetitions) {
		getMatrices(inputType1, inputType2);
		System.out.println("Sparsity: " + sparsity2 + "; rl: " + m + "; cl: " + n);

		TimingUtils.time(() -> sparseTest(binType, inputType1, inputType2), warmupRuns);
		TimingUtils.time(() -> denseTest(binType, inputType1, inputType2), warmupRuns);

		double[] sparseResults = TimingUtils.time(() -> sparseTest(binType, inputType1, inputType2), repetitions);
		double[] denseResults = TimingUtils.time(() -> denseTest(binType, inputType1, inputType2), repetitions);

		String sparseTime = TimingUtils.stats(sparseResults).split("\\+-")[0];
		String denseTime = TimingUtils.stats(denseResults).split("\\+-")[0];

		System.out.println("Sparse calculation: " + TimingUtils.stats(sparseResults));
		System.out.println("Dense calculation " + TimingUtils.stats(denseResults));
		return new String[] {sparseTime, denseTime};
	}

	private void sparseTest(BinType binType, InputType inputType1, InputType inputType2) {
		switch(binType) {
			case VECT_DIV_SCALAR -> {
				if((inputType1 == InputType.SCALAR)) {
					runSparseDivTest(inputType2, true);
				}
				else {
					runSparseDivTest(inputType1, false);
				}
			}
			case VECT_DIV -> {runSparseDivTest(inputType1, inputType2);}
		}
	}

	private void denseTest(BinType binType, InputType inputType1, InputType inputType2) {
		switch(binType) {
			case VECT_DIV_SCALAR -> {
				if((inputType1 == InputType.SCALAR)) {
					runDenseDivTest(inputType2, true);
				}
				else {
					runDenseDivTest(inputType1, false);
				}
			}
			case VECT_DIV -> {runDenseDivTest(inputType1, inputType2);}
		}
	}

	private void runSparseDivTest(InputType inputType1, InputType inputType2) {
		for(int j = 0; j < m; j++)
			vectDivWrite(n, sparseInA.values(j), sparseInB.values(j),
				sparseInA.indexes(j), sparseInB.indexes(j), sparseInA.pos(j),
				sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
	}

	private void runSparseDivTest(InputType inputType, boolean scalarVector) {
		if(scalarVector) {
			for(int j = 0; j < m; j++)
				vectDivWrite(n, scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j));
		}else {
			for(int j = 0; j < m; j++)
				vectDivWrite(n, sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j));
		}
	}

	private void runDenseDivTest(InputType inputType1, InputType inputType2) {
		for(int j = 0; j < m; j++)
			vectDivWrite(sparseInA.values(j), denseIn.values(j),
				sparseInA.indexes(j), sparseInA.pos(j), 0, sparseInA.size(j), n);
	}

	private void runDenseDivTest(InputType inputType, boolean scalarVector) {
		if(scalarVector) {
			for(int j = 0; j < m; j++)
				vectDivWrite(scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j), n);
		}else {
			for(int j = 0; j < m; j++)
				vectDivWrite(sparseInA.values(j), scalar,
					sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j), n);
		}
	}

	public void getMatrices(InputType inputType1, InputType inputType2) {
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		double[][] A = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityA, 1251);
		double[][] B = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityB, 532);
		MatrixBlock mA = DataConverter.convertToMatrixBlock(A);
		MatrixBlock mB = DataConverter.convertToMatrixBlock(B);
		MatrixBlock mC = DataConverter.convertToMatrixBlock(A);
		MatrixBlock mD = DataConverter.convertToMatrixBlock(B);

		if(inputType1 == InputType.SCALAR)
			scalar = mA.max();
		else if(inputType2 == InputType.SCALAR)
			scalar = mB.max();

		if(!mA.isInSparseFormat())
			mA.denseToSparse(true);
		sparseInA = new SparseBlockMCSR(mA.getSparseBlock());
		if(!mB.isInSparseFormat())
			mB.denseToSparse(true);
		sparseInB = new SparseBlockMCSR(mB.getSparseBlock());

		denseIn = mD.getDenseBlock();
	}
}
