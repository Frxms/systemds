package org.apache.sysds.performance.primitives;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;

import static org.apache.sysds.runtime.codegen.LibSpoofPrimitives.vectDivWrite;

public class PrimitivesTest {

	private final int m;
	private final int n;
	private final double sparsity2;
	private SparseBlockMCSR[] sparseM;
	private MatrixBlock[] denseM;

	private static final double sparsity1 = 0.9;

	public PrimitivesTest(int rl, int cl, double sparsity) {
		m = rl;
		n = cl;
		this.sparsity2 = sparsity;
	}

	public void sparseTest(BinType binType, InputType inputType1, InputType inputType2) {
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

	public void denseTest(BinType binType, InputType inputType1, InputType inputType2) {
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
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 1264);
		double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 1265);
		for(int j = 0; j < m; j++)
			vectDivWrite(n, inA.getSparseBlock().values(j), inB.getSparseBlock().values(j),
				inA.getSparseBlock().indexes(j), inB.getSparseBlock().indexes(j), inA.getSparseBlock().pos(j),
				inB.getSparseBlock().pos(j), inA.getSparseBlock().size(j), inB.getSparseBlock().size(j));
	}

	private void runSparseDivTest(InputType inputType, boolean scalarVector) {
		double sparsity = (inputType == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock in = MatrixBlock.randOperations(m, n, sparsity, -5, 5, "uniform", 1264);
		if(scalarVector) {
			for(int j = 0; j < m; j++)
				vectDivWrite(n, in.max(), in.getSparseBlock().values(j),
					in.getSparseBlock().indexes(j), in.getSparseBlock().pos(j), in.getSparseBlock().size(j));
		}else {
			for(int j = 0; j < m; j++)
				vectDivWrite(n, in.getSparseBlock().values(j), in.max(),
					in.getSparseBlock().indexes(j), in.getSparseBlock().pos(j), in.getSparseBlock().size(j));
		}
	}

	private void runDenseDivTest(InputType inputType1, InputType inputType2) {
		double sparsityA = (inputType1 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inA = MatrixBlock.randOperations(m, n, sparsityA, -5, 5, "uniform", 1264);
		double sparsityB = (inputType2 == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock inB = MatrixBlock.randOperations(m, n, sparsityB, -5, 5, "uniform", 1265);
		for(int j = 0; j < m; j++)
			vectDivWrite(inA.getSparseBlock().values(j), inB.getDenseBlockValues(),
				inA.getSparseBlock().indexes(j), inA.getSparseBlock().pos(j), 0, inA.getSparseBlock().size(j), n);
	}

	private void runDenseDivTest(InputType inputType, boolean scalarVector) {
		double sparsity = (inputType == InputType.VECTOR_DENSE) ? sparsity1 : sparsity2;
		MatrixBlock in = MatrixBlock.randOperations(m, n, sparsity, -5, 5, "uniform", 1264);
		if(scalarVector) {
			for(int j = 0; j < m; j++)
				vectDivWrite(in.max(), in.getSparseBlock().values(j),
					in.getSparseBlock().indexes(j), in.getSparseBlock().pos(j), in.getSparseBlock().size(j), n);
		}else {
			for(int j = 0; j < m; j++)
				vectDivWrite(in.getSparseBlock().values(j), in.max(),
					in.getSparseBlock().indexes(j), in.getSparseBlock().pos(j), in.getSparseBlock().size(j), n);
		}
	}

	public SparseBlockMCSR[] getSparseMatrices(double sparsityA, double sparsityB) {
		double[][] A = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityA, 1251);
		double[][] B = TestUtils.generateTestMatrix(m, n, -5, 5, sparsityB, 532);
		MatrixBlock mA = DataConverter.convertToMatrixBlock(A);
		MatrixBlock mB = DataConverter.convertToMatrixBlock(B);

		SparseBlockMCSR inA;
		SparseBlockMCSR inB;
		if(!mA.isInSparseFormat()) {
			mA.denseToSparse(true);
			inA = new SparseBlockMCSR(mA.getSparseBlock());
		} else {
			inA = new SparseBlockMCSR(mA.getSparseBlock());
		}
		if(!mB.isInSparseFormat()) {
			mB.denseToSparse(true);
			inB = new SparseBlockMCSR(mB.getSparseBlock());
		} else {
			inB = new SparseBlockMCSR(mB.getSparseBlock());
		}
		return new SparseBlockMCSR[] {inA, inB};
	}
}
