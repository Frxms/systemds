package org.apache.sysds.performance.primitives;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;
import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.*;

public class BinaryPrimitivesTest {

	private static final double sparsity1 = 0.9;

	private final double sparsity2;
	private final int m;
	private final int n;
	private final boolean branching;

	private SparseBlockMCSR sparseInA;
	private SparseBlockMCSR sparseInB;
	private DenseBlock denseIn;
	private double scalar;


	public BinaryPrimitivesTest(int rl, int cl, double sparsity, boolean branching) {
		m = rl;
		n = cl;
		this.sparsity2 = sparsity;
		this.branching = branching;
	}

	public String[] primitiveTester(BinType binType, InputType inputType1, InputType inputType2, int warmupRuns, int repetitions) {
		getMatrices(inputType1, inputType2);
		System.out.println("Sparsity: " + sparsity2 + "; rl: " + m + "; cl: " + n);

		setupThreadLocalMemory(1, n);
		setupSparseThreadLocalMemory(1, (int) (n*sparsity2 + 100*sparsity2), -1);

		TimingUtils.time(() -> sparseTest(binType, inputType1, inputType2), warmupRuns);
		TimingUtils.time(() -> denseTest(binType, inputType1, inputType2), warmupRuns);

		double[] sparseResults = TimingUtils.time(() -> sparseTest(binType, inputType1, inputType2), repetitions);
		double[] denseResults = TimingUtils.time(() -> denseTest(binType, inputType1, inputType2), repetitions);

		String sparseTime = TimingUtils.stats(sparseResults).split("\\+-")[0];
		String denseTime = TimingUtils.stats(denseResults).split("\\+-")[0];

		cleanupThreadLocalMemory();
		cleanupSparseThreadLocalMemory();

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
			case VECT_DIV -> {
				if(branching)
					runSparseDivBranchingTest();
				else
					runSparseDivTest();
			}
			case VECT_MULT_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runSparseMultTestSV();
				}else {
					runSparseMultTestVS();
				}
			}
			case VECT_MULT -> {runSparseMultTest();}
			case VECT_MIN_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runSparseMinTestSV();
				}else {
					runSparseMinTestVS();
				}
			}
			case VECT_MIN -> {runSparseMinTest();}
			case VECT_XOR_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runSparseXorTestSV();
				}else {
					runSparseXorTestVS();
				}
			}
			case VECT_XOR -> {runSparseXorTest();}
			case VECT_MINUS  -> runSparseMinusTest();
			case VECT_PLUS -> runSparsePlusTest();
			case VECT_POW_SCALAR -> runSparsePowTest();
			case VECT_NOTEQUAL_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runSparseNotequalTestSV();
				}else {
					runSparseNotequalTestVS();
				}
			}
			case VECT_NOTEQUAL -> runSparseNotequalTest();
			case VECT_LESS_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runSparseLessTestSV();
				}else {
					runSparseLessTestVS();
				}
			}
			case VECT_LESS -> runSparseLessTest();
			case VECT_EQUAL -> runSparseEqualTest();
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
			case VECT_MIN_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runDenseMinTestSV();
				}else {
					runDenseMinTestVS();
				}
			}
			case VECT_MIN -> {runDenseMinTest();}
			case VECT_XOR_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runDenseXorTestSV();
				}else {
					runDenseXorTestVS();
				}
			}
			case VECT_XOR -> {runDenseXorTest();}
			case VECT_MINUS  -> runDenseMinusTest();
			case VECT_PLUS -> runDensePlusTest();
			case VECT_POW_SCALAR -> runDensePowTest();
			case VECT_NOTEQUAL_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runDenseNotequalTestSV();
				}else {
					runDenseNotequalTestVS();
				}
			}
			case VECT_NOTEQUAL -> runDenseNotequalTest();
			case VECT_LESS_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					runDenseLessTestSV();
				}else {
					runDenseLessTestVS();
				}
			}
			case VECT_LESS -> runDenseLessTest();
			case VECT_EQUAL -> runSparseEqualTest();
		}
	}

	private void runSparseDivTest() {
		for(int i = 0; i < m; i++)
			vectDivWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	private void runSparseMultTest() {
		for(int i = 0; i < m; i++)
			vectMultWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	private void runSparseMinTest() {
		for(int i = 0; i < m; i++)
			vectMinWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	private void runSparseMinusTest() {
		for(int i = 0; i < m; i++)
			vectMinusWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	private void runSparsePlusTest() {
		for(int i = 0; i < m; i++)
			vectPlusWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	private void runSparseXorTest() {
		for(int i = 0; i < m; i++)
			vectXorWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	private void runSparseNotequalTest() {
		for(int i = 0; i < m; i++)
			vectNotequalWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	private void runSparseLessTest() {
		for(int i = 0; i < m; i++)
			vectLessWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	public void runSparseDivBranchingTest() {
		for(int i = 0; i < m; i++) {
			vectDivWriteB(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
		}
	}

	private void runSparseEqualTest() {
		for(int i = 0; i < m; i++)
			vectEqualWrite(n,
				sparseInA.values(i), sparseInB.values(i), sparseInA.indexes(i), sparseInB.indexes(i),
				sparseInA.pos(i), sparseInB.pos(i), sparseInA.size(i), sparseInB.size(i));
	}

	private void runSparseDivTestSV() {
		for(int i = 0; i < m; i++)
			vectDivWrite(n, scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i));
	}

	private void runSparseMultTestSV() {
		for(int i = 0; i < m; i++)
			vectMultWrite(n, scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i));
	}

	private void runSparseMinTestSV() {
		for(int i = 0; i < m; i++)
			vectMinWrite(n, scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i));
	}

	private void runSparseXorTestSV() {
		for(int i = 0; i < m; i++)
			vectXorWrite(n, scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i));
	}

	private void runSparseNotequalTestSV() {
		for(int i = 0; i < m; i++)
			vectNotequalWrite(n, scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i));
	}

	private void runSparseLessTestSV() {
		for(int i = 0; i < m; i++)
			vectLessWrite(n, scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i));
	}

	public void runSparseDivTestVS() {
		for(int i = 0; i < m; i++)
			vectDivWrite(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
	}

	private void runSparseMultTestVS() {
		for(int i = 0; i < m; i++)
			vectMultWrite(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
	}

	private void runSparseMinTestVS() {
		for(int i = 0; i < m; i++)
			vectMinWrite(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
	}

	private void runSparseXorTestVS() {
		for(int i = 0; i < m; i++)
			vectXorWrite(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
	}

	private void runSparseNotequalTestVS() {
		for(int i = 0; i < m; i++)
			vectNotequalWrite(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
	}

	private void runSparseLessTestVS() {
		for(int i = 0; i < m; i++)
			vectLessWrite(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
	}

	private void runSparsePowTest() {
		for(int i = 0; i < m; i++)
			vectPowWrite(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
	}

	private void runDenseDivTest() {
		for(int i = 0; i < m; i++)
			vectDivWrite(sparseInA.values(i), denseIn.values(i),
				sparseInA.indexes(i), sparseInA.pos(i), 0, sparseInA.size(i), n);
	}

	private void runDenseMultTest() {
		for(int i = 0; i < m; i++)
			vectMultWrite(sparseInA.values(i), denseIn.values(i),
				sparseInA.indexes(i), sparseInA.pos(i), 0, sparseInA.size(i), n);
	}

	private void runDenseMinTest() {
		for(int i = 0; i < m; i++)
			vectMinWrite(sparseInA.values(i), denseIn.values(i),
				sparseInA.indexes(i), sparseInA.pos(i), 0, sparseInA.size(i), n);
	}

	private void runDenseMinusTest() {
		for(int i = 0; i < m; i++)
			vectMinusWrite(sparseInA.values(i), denseIn.values(i),
				sparseInA.indexes(i), sparseInA.pos(i), 0, sparseInA.size(i), n);
	}

	private void runDensePlusTest() {
		for(int i = 0; i < m; i++)
			vectPlusWrite(sparseInA.values(i), denseIn.values(i),
				sparseInA.indexes(i), sparseInA.pos(i), 0, sparseInA.size(i), n);
	}

	private void runDenseXorTest() {
		for(int i = 0; i < m; i++)
			vectXorWrite(sparseInA.values(i), denseIn.values(i),
				sparseInA.indexes(i), sparseInA.pos(i), 0, sparseInA.size(i), n);
	}

	private void runDenseNotequalTest() {
		for(int i = 0; i < m; i++)
			vectNotequalWrite(sparseInA.values(i), denseIn.values(i),
				sparseInA.indexes(i), sparseInA.pos(i), 0, sparseInA.size(i), n);
	}

	private void runDenseLessTest() {
		for(int i = 0; i < m; i++)
			vectLessWrite(sparseInA.values(i), denseIn.values(i),
				sparseInA.indexes(i), sparseInA.pos(i), 0, sparseInA.size(i), n);
	}

	private void runDenseDivTestSV() {
		for(int i = 0; i < m; i++)
			vectDivWrite(scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i), n);
	}

	private void runDenseMinTestSV() {
		for(int i = 0; i < m; i++)
			vectMinWrite(scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i), n);
	}

	private void runDenseXorTestSV() {
		for(int i = 0; i < m; i++)
			vectXorWrite(scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i), n);
	}

	private void runDenseNotequalTestSV() {
		for(int i = 0; i < m; i++)
			vectNotequalWrite(scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i), n);
	}

	private void runDenseLessTestSV() {
		for(int i = 0; i < m; i++)
			vectLessWrite(scalar, sparseInB.values(i), sparseInB.indexes(i), sparseInB.pos(i), sparseInB.size(i), n);
	}

	private void runDenseDivTestVS() {
		for(int i = 0; i < m; i++)
			vectDivWrite(sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i), n);
	}

	private void runDenseMultTestVS() {
		for(int i = 0; i < m; i++)
			vectMultWrite(sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i), n);
	}

	private void runDenseMinTestVS() {
		for(int i = 0; i < m; i++)
			vectMultWrite(sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i), n);
	}

	private void runDenseXorTestVS() {
		for(int i = 0; i < m; i++)
			vectXorWrite(sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i), n);
	}

	private void runDenseNotequalTestVS() {
		for(int i = 0; i < m; i++)
			vectNotequalWrite(sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i), n);
	}

	private void runDenseLessTestVS() {
		for(int i = 0; i < m; i++)
			vectLessWrite(sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i), n);
	}

	private void runDensePowTest() {
		for(int i = 0; i < m; i++)
			vectPowWrite(sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i), n);
	}

	public void getMatrices(InputType inputType1, InputType inputType2) {
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		double[][] A = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityA, 1251);
		double[][] B = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityB, 532);
		double[][] D = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityA, 1251);
		MatrixBlock mA = DataConverter.convertToMatrixBlock(A);
		MatrixBlock mB = DataConverter.convertToMatrixBlock(B);
		MatrixBlock mD = DataConverter.convertToMatrixBlock(D);

		if(inputType1 == InputType.SCALAR)
			scalar = mA.max();
		else if(inputType2 == InputType.SCALAR)
			scalar = mB.max();
//		scalar = -1;

		if(!mA.isInSparseFormat())
			mA.denseToSparse(true);
		sparseInA = new SparseBlockMCSR(mA.getSparseBlock());
		if(!mB.isInSparseFormat())
			mB.denseToSparse(true);
		sparseInB = new SparseBlockMCSR(mB.getSparseBlock());

		if(mD.isInSparseFormat())
			mD.sparseToDense();
		denseIn = mD.getDenseBlock();
	}
}
