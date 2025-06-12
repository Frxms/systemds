package org.apache.sysds.performance.primitives;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class SparseRowPerfTest {

	private final int m;
	private final int n;
	private final int warmupRuns;
	private final int repetitions;
	private final int testSize;
	private final double maxSparsity;

	public SparseRowPerfTest() {
		this(1000, 10000, 100, 1000, 0.1, 5);
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
		double[] sparsityVals = sparsityValues(false, true);
		String[] sparseResults = new String[testSize];
		String[] denseResults = new String[testSize];
		for(int k = 0; k < testSize; k++) {
			PrimitivesTest tester = new PrimitivesTest(m, n, sparsityVals[k]);
			String[] results = tester.primitiveTester(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR, warmupRuns, repetitions);
			sparseResults[k] = results[0];
			denseResults[k] = results[1];
		}
		logResults(sparsityVals, sparseResults, true);
		logResults(sparsityVals, denseResults, false);
	}

	private double[] sparsityValues(boolean exp, boolean linear) {
		double[] sparsity = new double[testSize];
		double currVal = maxSparsity;
		if(exp) {
			for(int i = 1; i < testSize; i++) {
				sparsity[i-1] = currVal;
				currVal = currVal * Math.exp(-1*i);
			}
			sparsity[testSize-1] = currVal * Math.exp(-1*testSize-1);
		}else if(linear){
			for(int i = 0; i < testSize; i++) {
				sparsity[i] = currVal;
				currVal -= 0.01;
			}
		} else {
			for(int i = 0; i < testSize; i++) {
				sparsity[i] = currVal;
				currVal /= 3;
			}
		}
		return sparsity;
	}

	public void logResults(double[] sparsityVals, String[] result, boolean sparse) {
		String fileName = sparse ? "linearGrading_sparse_" : "linearGrading_dense_";
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(new FileWriter("C:\\Users\\tomok\\OneDrive - Technische Universität Berlin\\Bachelorarbeit\\performance\\results\\"
				+ fileName + BinType.VECT_DIV_SCALAR.name() + ".csv"));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		writer.printf("Repetitions: %1$2s, rl: %2$2s, cl: %3$2s%n", repetitions, m, n);
		writer.printf("%1$2s;%2$2s%n", "Sparsity", "time in ms");
		for(int i = 0; i < testSize; i++) {
			writer.printf("%.3f;%2$2s%n", sparsityVals[i], result[i]);
		}
		writer.flush();
		writer.close();
	}

	public static void main(String[] args) {
		new SparseRowPerfTest().testPrimitivePerf();
	}
}
