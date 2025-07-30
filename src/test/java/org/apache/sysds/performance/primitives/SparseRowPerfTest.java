package org.apache.sysds.performance.primitives;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary.BinType;
import org.apache.sysds.hops.codegen.cplan.CNodeUnary.UnaryType;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest.InputType;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class SparseRowPerfTest {

	private final int m;
	private final int n;
	private final int warmupRuns;
	private final int repetitions;
	private final double maxSparsity;
	private int testSize;
	private SparsityType sparsityType;
	double[] sparsityVals;
	int[] cols;
	int[] rows;

	public enum TestType{HYBRID, SPARSITY, MATRIX, B_HYBRID, B_SPARSITY, B_MATRIX}

	public enum SparsityType{GEO, LIN, DIV, SET}

	public SparseRowPerfTest() {
		this(5000, 10000, 100, 2500, 1, 7);
	}

	public SparseRowPerfTest(int rl, int cl, int warmupRuns, int repetitions, double sparsity, int testSize) {
		m = rl;
		n = cl;
		this.warmupRuns = warmupRuns;
		this.repetitions = repetitions;
		this.maxSparsity = sparsity;
		this.testSize = testSize;
	}

	public void testBinaryPrimitivePerf(BinType binType, InputType input1, InputType input2, TestType testType, SparsityType sparsityType) {
		chooseTestType(testType, sparsityType);
		String[] sparseResults = new String[testSize*3];
		String[] denseResults = new String[testSize*3];
		BinaryPrimitivesTest tester = new BinaryPrimitivesTest();
		if(testType == TestType.SPARSITY)
			for(int k = 0; k < testSize; k++) {
				String[] results = tester.primitiveTester(m, n, sparsityVals[k], false, binType, input1, input2, warmupRuns, repetitions);
				sparseResults[k] = results[0];
				denseResults[k] = results[1];
			}
		else if(testType == TestType.MATRIX)
			for(int k = 0; k < 3; k++) {
				String[] results = tester.primitiveTester(rows[k], cols[k], maxSparsity, false, binType, input1, input2, warmupRuns, repetitions);
				sparseResults[k] = results[0];
				denseResults[k] = results[1];
			}
		else if(testType == TestType.B_SPARSITY)
			for(int k = 0; k < testSize; k++) {
				String[] results = tester.primitiveTester(m, n, sparsityVals[k], true, binType, input1, input2, warmupRuns, repetitions);
				sparseResults[k] = results[0];
				denseResults[k] = results[1];
			}
		else if(testType == TestType.B_MATRIX)
			for(int k = 0; k < 3; k++) {
				String[] results = tester.primitiveTester(rows[k], cols[k], maxSparsity, true, binType, input1, input2, warmupRuns, repetitions);
				sparseResults[k] = results[0];
				denseResults[k] = results[1];
			}
		else if(testType == TestType.HYBRID) {
			int resIncr = 0;
			for(int k = 0; k < 3; k++) {
				for(int l = 0; l < testSize; l++) {
					String[] results = tester.primitiveTester(rows[k], cols[k], sparsityVals[l], false, binType, input1, input2, warmupRuns, repetitions);
					sparseResults[resIncr] = results[0];
					denseResults[resIncr] = results[1];
					resIncr++;
				}
			}
		} else if(testType == TestType.B_HYBRID) {
			int resIncr = 0;
			for(int k = 0; k < 3; k++) {
				for(int l = 0; l < testSize; l++) {
					String[] results = tester.primitiveTester(rows[k], cols[k], sparsityVals[l], false, binType, input1,
						input2, warmupRuns, repetitions);
					sparseResults[resIncr] = results[0];
					denseResults[resIncr] = results[1];
				}
			}
		} else
				System.out.println("no matching TestType found");

		logResults(testType, sparseResults, true, binType);
		logResults(testType, denseResults, false, binType);
	}

	public void testUnaryPrimitivePerf(UnaryType uType, InputType input1) {
		double[] sparsityVals = sparsityValues(SparsityType.DIV);
		String[] sparseResults = new String[testSize];
		String[] denseResults = new String[testSize];
		for(int k = 0; k < testSize; k++) {
			UnaryPrimitivesTest tester = new UnaryPrimitivesTest(m, n, sparsityVals[k]);
			String[] results = tester.primitiveTester(uType, input1, warmupRuns, repetitions);
			sparseResults[k] = results[0];
			denseResults[k] = results[1];
		}
		logResults(sparsityVals, sparseResults, true, uType);
		logResults(sparsityVals, denseResults, false, uType);
	}

	private void chooseTestType(TestType testType, SparsityType sparsityType) {
		if(testType == TestType.B_SPARSITY || testType == TestType.SPARSITY) {
			sparsityVals = sparsityValues(sparsityType);
		} else if (testType == TestType.B_MATRIX || testType == TestType.MATRIX){
			cols = colsValues();
			rows = rowsValues();
		} else if(testType == TestType.B_HYBRID || testType == TestType.HYBRID) {
			sparsityVals = sparsityValues(sparsityType);
			cols = colsValues();
			rows = rowsValues();
		}
	}

	private double[] sparsityValues(SparsityType sparsityType) {
		double[] sparsity = new double[testSize];
		double currVal = maxSparsity;
		if(sparsityType == SparsityType.GEO) {
			for(int i = 1; i < testSize; i++) {
				sparsity[i-1] = currVal;
				currVal = currVal * Math.exp(-1*i);
			}
			sparsity[testSize-1] = currVal * Math.exp(-1*testSize-1);
		}else if(sparsityType == SparsityType.LIN){
			for(int i = 0; i < testSize; i++) {
				sparsity[i] = currVal;
				currVal -= 0.01;
			}
		} else if(sparsityType == SparsityType.DIV) {
			for(int i = 0; i < testSize; i++) {
				sparsity[i] = currVal;
				currVal /= 3;
			}
		} else if(sparsityType == SparsityType.SET) {
			sparsity = new double[] {1, 0.3333, 0.1111, 0.0333, 0.0111, 0.0033, 0.0011};
			testSize = 7;
		}
		return sparsity;
	}

	private int[] rowsValues() {
		int[] rows = new int[3];
		rows[0] = m;
		rows[1] = (int) Math.sqrt(m*n);
		rows[2] = n;
		return rows;
	}

	private int[] colsValues() {
		int[] cols = new int[3];
		cols[0] = n;
		cols[1] = (int) Math.sqrt(m*n);
		cols[2] = m;
		return cols;
	}

	public void logResults(TestType testType, String[] result, boolean sparse, BinType binType) {
		String currDate = LocalDateTime.now().format(DateTimeFormatter.ofPattern("ddMMyyyy_HHmmss"));
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(new FileWriter("C:\\Users\\tomok\\OneDrive - Technische Universität Berlin\\Bachelorarbeit\\performance\\results\\"
				+ currDate + "_" + (sparse ? "sparse" : "dense") + "_" + binType.name() + ".csv"));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		writer.printf("%4$s Repetitions: %1$2s, rl: %2$2s, cl: %3$2s%n", repetitions, m, n, binType.name());
		writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", "Sparsity", "rows", "cols","time in ms");
		if(testType == TestType.B_SPARSITY || testType == TestType.SPARSITY) {
			for(int i = 0; i < testSize; i++) {
				writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", sparsityVals[i], m, n, result[i]);
			}
		} else if(testType == TestType.B_MATRIX || testType == TestType.MATRIX){
			for(int i = 0; i < 3; i++) {
				writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", maxSparsity, rows[i], cols[i], result[i]);
			}
		} else if(testType == TestType.B_HYBRID || testType == TestType.HYBRID){
			int incr = 0;
			for(int i = 0; i < 3; i++) {
				for(int j = 0; j < testSize; j++) {
					writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", sparsityVals[j], rows[i], cols[i], result[incr]);
					incr++;
				}
			}
		}

		writer.flush();
		writer.close();
	}

	public void logResults(double[] sparsityVals, String[] result, boolean sparse, UnaryType uType) {
		String currDate = LocalDateTime.now().format(DateTimeFormatter.ofPattern("ddMMyyyy_HHmmss"));
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(new FileWriter("C:\\Users\\tomok\\OneDrive - Technische Universität Berlin\\Bachelorarbeit\\performance\\results\\"
				+ currDate + "_" + (sparse ? "sparse" : "dense") + "_" + sparsityType + "_" + uType.name() + ".csv"));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		writer.printf("%4$s Repetitions: %1$2s, rl: %2$2s, cl: %3$2s%n", repetitions, m, n, uType.name());
		writer.printf("%1$2s;%2$2s%n", "Sparsity", "time in ms");
		for(int i = 0; i < testSize; i++) {
			writer.printf("%.3f;%2$2s%n", sparsityVals[i], result[i]);
		}
		writer.flush();
		writer.close();
	}

	public static void main(String[] args) {
//		new SparseRowPerfTest().testBinaryPrimitivePerf(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR, TestType.HYBRID, SparsityType.SET);
//		new SparseRowPerfTest().testBinaryPrimitivePerf(BinType.VECT_DIV_SCALAR, InputType.SCALAR, InputType.VECTOR_SPARSE, TestType.HYBRID, SparsityType.SET);
		new SparseRowPerfTest().testBinaryPrimitivePerf(BinType.VECT_EQUAL, InputType.VECTOR_SPARSE, InputType.VECTOR_SPARSE, TestType.HYBRID, SparsityType.SET);


//		new SparseRowPerfTest().testUnaryPrimitivePerf(UnaryType.VECT_SQRT, InputType.VECTOR_SPARSE);
	}
}
