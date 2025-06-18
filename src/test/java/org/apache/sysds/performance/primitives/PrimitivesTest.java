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
			case VECT_DIV -> {runSparseDivTest();}
			case VECT_MULT_SCALAR -> {
				if(inputType1 == InputType.SCALAR) {
					break;
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
		}
	}

	private void runSparseDivTest() {
		for(int j = 0; j < m; j++)
			vectDivWrite(n,
				sparseInA.values(j), sparseInB.values(j), sparseInA.indexes(j), sparseInB.indexes(j),
				sparseInA.pos(j), sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
	}

	private void runSparseMultTest() {
		for(int j = 0; j < m; j++)
			vectMultWrite(n,
				sparseInA.values(j), sparseInB.values(j), sparseInA.indexes(j), sparseInB.indexes(j),
				sparseInA.pos(j), sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
	}

	private void runSparseMinTest() {
		for(int j = 0; j < m; j++)
			vectMinWrite(n,
				sparseInA.values(j), sparseInB.values(j), sparseInA.indexes(j), sparseInB.indexes(j),
				sparseInA.pos(j), sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
	}

	private void runSparseMinusTest() {
		for(int j = 0; j < m; j++)
			vectMinusWrite(n,
				sparseInA.values(j), sparseInB.values(j), sparseInA.indexes(j), sparseInB.indexes(j),
				sparseInA.pos(j), sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
	}

	private void runSparsePlusTest() {
		for(int j = 0; j < m; j++)
			vectPlusWrite(n,
				sparseInA.values(j), sparseInB.values(j), sparseInA.indexes(j), sparseInB.indexes(j),
				sparseInA.pos(j), sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
	}

	private void runSparseXorTest() {
		for(int j = 0; j < m; j++)
			vectXorWrite(n,
				sparseInA.values(j), sparseInB.values(j), sparseInA.indexes(j), sparseInB.indexes(j),
				sparseInA.pos(j), sparseInB.pos(j), sparseInA.size(j), sparseInB.size(j));
	}

	private void runSparseDivTestSV() {
		for(int j = 0; j < m; j++)
			vectDivWrite(n, scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j));
	}

	private void runSparseMinTestSV() {
		for(int j = 0; j < m; j++)
			vectMinWrite(n, scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j));
	}

	private void runSparseXorTestSV() {
		for(int j = 0; j < m; j++)
			vectXorWrite(n, scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j));
	}

	public void runSparseDivTestVS() {
		for(int j = 0; j < m; j++)
			vectDivWrite(n, sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j));
	}

	private void runSparseMultTestVS() {
		for(int j = 0; j < m; j++)
			vectMultWrite(n, sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j));
	}

	private void runSparseMinTestVS() {
		for(int j = 0; j < m; j++)
			vectMinWrite(n, sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j));
	}

	private void runSparseXorTestVS() {
		for(int j = 0; j < m; j++)
			vectXorWrite(n, sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j));
	}

	private void runSparsePowTest() {
		for(int j = 0; j < m; j++)
			vectPowWrite(n, sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j));
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

	private void runDenseMinTest() {
		for(int j = 0; j < m; j++)
			vectMinWrite(sparseInA.values(j), denseIn.values(j),
				sparseInA.indexes(j), sparseInA.pos(j), 0, sparseInA.size(j), n);
	}

	private void runDenseMinusTest() {
		for(int j = 0; j < m; j++)
			vectMinusWrite(sparseInA.values(j), denseIn.values(j),
				sparseInA.indexes(j), sparseInA.pos(j), 0, sparseInA.size(j), n);
	}

	private void runDensePlusTest() {
		for(int j = 0; j < m; j++)
			vectPlusWrite(sparseInA.values(j), denseIn.values(j),
				sparseInA.indexes(j), sparseInA.pos(j), 0, sparseInA.size(j), n);
	}

	private void runDenseXorTest() {
		for(int j = 0; j < m; j++)
			vectXorWrite(sparseInA.values(j), denseIn.values(j),
				sparseInA.indexes(j), sparseInA.pos(j), 0, sparseInA.size(j), n);
	}

	private void runDenseDivTestSV() {
		for(int j = 0; j < m; j++)
			vectDivWrite(scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j), n);
	}

	private void runDenseMinTestSV() {
		for(int j = 0; j < m; j++)
			vectMinWrite(scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j), n);
	}

	private void runDenseXorTestSV() {
		for(int j = 0; j < m; j++)
			vectXorWrite(scalar, sparseInB.values(j), sparseInB.indexes(j), sparseInB.pos(j), sparseInB.size(j), n);
	}

	private void runDenseDivTestVS() {
		for(int j = 0; j < m; j++)
			vectDivWrite(sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j), n);
	}

	private void runDenseMultTestVS() {
		for(int j = 0; j < m; j++)
			vectMultWrite(sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j), n);
	}

	private void runDenseMinTestVS() {
		for(int j = 0; j < m; j++)
			vectMultWrite(sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j), n);
	}

	private void runDenseXorTestVS() {
		for(int j = 0; j < m; j++)
			vectXorWrite(sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j), n);
	}

	private void runDensePowTest() {
		for(int j = 0; j < m; j++)
			vectPowWrite(sparseInA.values(j), scalar, sparseInA.indexes(j), sparseInA.pos(j), sparseInA.size(j), n);
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
//		scalar = 0;

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
