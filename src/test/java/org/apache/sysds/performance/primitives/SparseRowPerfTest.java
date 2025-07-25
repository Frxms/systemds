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
	private final int testSize;
	private final double maxSparsity;
	private String sparsityType;
	double[] sparsityVals;
	int[] cols;
	int[] rows;
	boolean testType;

	public SparseRowPerfTest() {
		this(2000, 4000, 100, 2500, 0.1, 3, false);
	}

	public SparseRowPerfTest(int rl, int cl, int warmupRuns, int repetitions, double sparsity, int testSize, boolean testType) {
		m = rl;
		n = cl;
		this.warmupRuns = warmupRuns;
		this.repetitions = repetitions;
		this.maxSparsity = sparsity;
		this.testSize = testSize;
		this.testType = testType;
	}

	private void testBinaryPrimitivePerf(BinType binType, InputType input1, InputType input2) {
		chooseTestType(testType);
		String[] sparseResults = new String[testSize];
		String[] denseResults = new String[testSize];
		for(int k = 0; k < testSize; k++) {
			BinaryPrimitivesTest tester;
			if(testType)
				 tester = new BinaryPrimitivesTest(m, n, sparsityVals[k]);
			else
				tester = new BinaryPrimitivesTest(rows[k], cols[k], maxSparsity);

			String[] results = tester.primitiveTester(binType, input1, input2, warmupRuns, repetitions);
			sparseResults[k] = results[0];
			denseResults[k] = results[1];
		}
//		logResults(testType, sparseResults, true, binType);
//		logResults(testType, denseResults, false, binType);
	}

	public void testUnaryPrimitivePerf(UnaryType uType, InputType input1) {
		double[] sparsityVals = sparsityValues(false, true);
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

	private void chooseTestType(boolean type) {
		if(type) {
			sparsityVals = sparsityValues(false, false);
		} else {
			cols = colsValues();
			rows = rowsValues();
		}
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
			sparsityType = "exp";
		}else if(linear){
			for(int i = 0; i < testSize; i++) {
				sparsity[i] = currVal;
				currVal -= 0.01;
			}
			sparsityType = "lin";
		} else {
			for(int i = 0; i < testSize; i++) {
				sparsity[i] = currVal;
				currVal /= 3;
			}
			sparsityType = "geo";
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

	public void logResults(boolean type, String[] result, boolean sparse, BinType binType) {
		String currDate = LocalDateTime.now().format(DateTimeFormatter.ofPattern("ddMMyyyy_HHmmss"));
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(new FileWriter("C:\\Users\\tomok\\OneDrive - Technische Universität Berlin\\Bachelorarbeit\\performance\\results\\"
				+ currDate + "_" + (sparse ? "sparse" : "dense") + "_" + sparsityType + "_" + binType.name() + ".csv"));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		writer.printf("%4$s Repetitions: %1$2s, rl: %2$2s, cl: %3$2s%n", repetitions, m, n, binType.name());
		writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", "Sparsity", "rows", "cols","time in ms");
		if(type) {
			for(int i = 0; i < testSize; i++) {
				writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", sparsityVals[i], m, n, result[i]);
			}
		} else {
			for(int i = 0; i < testSize; i++) {
				writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", maxSparsity, rows[i], cols[i], result[i]);
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
		new SparseRowPerfTest().testBinaryPrimitivePerf(BinType.VECT_MULT_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
		new SparseRowPerfTest().testBinaryPrimitivePerf(BinType.VECT_DIV_SCALAR, InputType.VECTOR_SPARSE, InputType.SCALAR);
//		new SparseRowPerfTest().testUnaryPrimitivePerf(UnaryType.VECT_SQRT, InputType.VECTOR_SPARSE);
	}
}
