package org.apache.sysds.performance.primitives;

import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.codegen.CPlanVectorPrimitivesTest;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class SparseRowAllocPerfTest {

	private static int m;
	private static int n;

	private static SparseBlockMCSR sparseInA;
	private SparseBlockMCSR sparseInB;
	private DenseBlock denseIn;
	private static double scalar;
	static int repetitions;
	static int warmupRuns;
	static int[] mVar = {100};
	static int[] nVar = {10000000, 1000000, 100000, 10000, 1000, 100, 10};

	public SparseRowAllocPerfTest() {
		this.m = 30;
		this.n = 10000;
		this.warmupRuns = 100;
		this.repetitions = 2500;
	}

	public static void main(String[] args) {
		SparseRowAllocPerfTest.testAlloc(3, 1000);
	}

	public void testDenAndSpaAlloc() {
			String[] results;
			String[] sparseRes = new String[nVar.length];
			String[] denseRes = new String[nVar.length];
			for(int i = 0; i < nVar.length; i++) {
				getMatrices(nVar[i]);
				results = testAlloc(2000, nVar[i]);
				sparseRes[i] = results[0];
				denseRes[i] = results[1];
			}
			logResults(sparseRes, true);
			logResults(denseRes, false);
	}

	public void compareInitAndAlloc() {
		String[] results;
		String[] sparseRes = new String[nVar.length];
		String[] denseRes = new String[nVar.length];
		for(int i = 0; i < nVar.length; i++) {
			getMatrices(nVar[i]);
			results = testAlloc(2000, nVar[i]);
			sparseRes[i] = results[0];
			denseRes[i] = results[1];
		}
		logResults(sparseRes, true);
		logResults(denseRes, false);
	}

	public static String[] compareAlloc(int numVectors, int len) {
		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len, -1);

		TimingUtils.time(() -> LibSpoofPrimitives.allocSparseVector(len), warmupRuns);
		double[] allocResults = TimingUtils.time(() -> LibSpoofPrimitives.allocSparseVector(len), repetitions);

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();

		LibSpoofPrimitives.setupThreadLocalMemory(numVectors, len);

		TimingUtils.time(() -> LibSpoofPrimitives.allocVector(len, false), warmupRuns);
		double[] initResults = TimingUtils.time(() -> LibSpoofPrimitives.allocVector(len, false), warmupRuns);

		LibSpoofPrimitives.cleanupThreadLocalMemory();

		String sparseTime = TimingUtils.stats(allocResults).split("\\+-")[0];
		String denseTime = TimingUtils.stats(initResults).split("\\+-")[0];

		System.out.println("Allocation test: " + TimingUtils.stats(allocResults));
		System.out.println("Initialization test: " + TimingUtils.stats(initResults));

		return new String[] {sparseTime, denseTime};
	}

	public static String[] testAlloc(int numVectors, int len) {

		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len, -1);

		TimingUtils.time(() -> allocateVector(len), warmupRuns);
		double[] allocResults = TimingUtils.time(() -> allocateVector(len), repetitions);

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();

		TimingUtils.time(() -> initializeVector(len), warmupRuns);
		double[] initResults = TimingUtils.time(() -> initializeVector(len), warmupRuns);

		String sparseTime = TimingUtils.stats(allocResults).split("\\+-")[0];
		String denseTime = TimingUtils.stats(initResults).split("\\+-")[0];

		System.out.println("Allocation test: " + TimingUtils.stats(allocResults));
		System.out.println("Initialization test: " + TimingUtils.stats(initResults));

		return new String[] {sparseTime, denseTime};
	}



	public static void allocateVector(int len) {
		SparseRowVector c = LibSpoofPrimitives.allocSparseVector(len);
		double[] a = sparseInA.values(12);
		int[] aix = sparseInA.indexes(12);
		int ai = sparseInA.pos(12);
		int alen = sparseInA.size(12);
		double bval = scalar;
		if( a == null ) {
			System.out.println("change row");
			return;
		}
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = a[j]*bval;
		}
	}

	public static void initializeVector(int len) {
		SparseRowVector c = new SparseRowVector(len);
		double[] a = sparseInA.values(12);
		int[] aix = sparseInA.indexes(12);
		int ai = sparseInA.pos(12);
		int alen = sparseInA.size(12);
		double bval = scalar;
		if( a == null ) {
			System.out.println("change row");
			return;
		}
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = a[j]*bval;
		}
	}

	public void getMatrices(int n) {
		double[][] A = TestUtils.generateTestMatrix(m, n, -5, 5, 0.2, 1251);
		double[][] B = TestUtils.generateTestMatrix(m, n, -5, 5, 0.2, 1345);
		MatrixBlock mA = DataConverter.convertToMatrixBlock(A);
		MatrixBlock mB = DataConverter.convertToMatrixBlock(B);

		scalar = mA.max();


		if(!mA.isInSparseFormat())
			mA.denseToSparse(true);
		sparseInA = new SparseBlockMCSR(mA.getSparseBlock());
		if(!mB.isInSparseFormat())
			mB.denseToSparse(true);
		sparseInB = new SparseBlockMCSR(mB.getSparseBlock());
	}

	public static void logResults(String[] result, boolean sparse) {
		String currDate = LocalDateTime.now().format(DateTimeFormatter.ofPattern("ddMMyyyy_HHmmss"));
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(new FileWriter("C:\\Users\\tomok\\OneDrive - Technische Universität Berlin\\Bachelorarbeit\\performance\\results\\"
				+ "allocTest" + currDate + "_" + (sparse ? "sparse" : "dense") + "_" + ".csv"));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		writer.printf("Repetitions: %1$2s, rl: %2$2s, cl: %3$2s%n", repetitions, m, n);
		writer.printf("%1$2s;%2$2s;%3$2s%n", "rows", "cols","time in ms");
		for(int i = 0; i < nVar.length; i++) {
			writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", m, n, result[i]);
		}

		writer.flush();
		writer.close();
	}
}
