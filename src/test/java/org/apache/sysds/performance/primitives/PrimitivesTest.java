package org.apache.sysds.performance.primitives;

import com.esotericsoftware.kryo.io.Input;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;
import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.*;

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
				if(inputType1 == InputType.SCALAR) {
					runSparseDivTestSV();
				}else {
					runSparseDivTestVS();
				}
			}
			case VECT_DIV -> {runSparseDivTest();}
			case VECT_MULT_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					break;
				}else {
					runSparseMultTestVS();
				}
			}
			case VECT_MULT -> {runSparseMultTest();}
		}
	}

	private void denseTest(BinType binType, InputType inputType1, InputType inputType2) {
		switch(binType) {
			case VECT_DIV_SCALAR -> {
				if((inputType1 == InputType.SCALAR)) {
					runDenseDivTestSV();
				}
				else {
					runDenseDivTestVS();
				}
			}
			case VECT_DIV -> {runDenseDivTest();}
			case VECT_MULT_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					break;
				} else {
					runDenseMultTestVS();
				}
			}
			case VECT_MULT -> {runDenseMultTest();}
		}
	}

	private void runSparseDivTest() {
		for(int j = 0; j < m; j++)
			vectDivWrite(n, sparseInA.values(j), sparseInB.values(j),
				sparseInA.indexes(j), sparseInB.indexes(j), sparseInA.pos(j),
				sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
	}

	private void runSparseMultTest() {
		for(int j = 0; j < m; j++) {
			vectMultWrite(n, sparseInA.values(j), sparseInB.values(j),
				sparseInA.indexes(j), sparseInB.indexes(j), sparseInA.pos(j),
				sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
		}
	}

	private void runSparseDivTestSV() {
		for(int j = 0; j < m; j++)
			vectDivWrite(n, scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j));
	}

	public void runSparseDivTestVS() {
		for(int j = 0; j < m; j++)
			vectDivWrite(n, sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j));
	}

	private void runSparseMultTestVS() {
		for(int j = 0; j < m; j++)
			vectMultWrite(n, sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j));
	}

	private void runDenseDivTest() {
		for(int j = 0; j < m; j++)
			vectDivWrite(sparseInA.values(j), denseIn.values(j),
				sparseInA.indexes(j), sparseInA.pos(j), 0, sparseInA.size(j), n);
	}

	private void runDenseMultTest() {
		for(int j = 0; j < m; j++)
			vectMultWrite(sparseInA.values(j), denseIn.values(j),
				sparseInA.indexes(j), sparseInA.pos(j), 0, sparseInA.size(j), n);
	}

	private void runDenseDivTestSV() {
		for(int j = 0; j < m; j++)
			vectDivWrite(scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j), n);
	}

	private void runDenseDivTestVS() {
		for(int j = 0; j < m; j++)
			vectDivWrite(sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j), n);
	}

	private void runDenseMultTestVS() {
		for(int j = 0; j < m; j++) {
			vectMultWrite(sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j), n);
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
