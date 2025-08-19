package org.apache.sysds.performance.primitives;

import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class SparseRowAllocPerfTest {

	private int m;
	private int n;

	private static SparseBlockMCSR sparseInA;
	private SparseBlockMCSR sparseInB;
	private DenseBlock denseIn;
	private static double scalar;
	int repetitions;
	int warmupRuns;
	int[] mVar = {100};
	int[] nVar = {100000, 10000, 1000, 100, 10};
	double[] sparsities = {1, 0.3333, 0.1111, 0.0333, 0.0111, 0.0033, 0.0011};
//	double sparsity = 1;

	public SparseRowAllocPerfTest() {
		this.m = 30;
		this.n = 10000;
		this.warmupRuns = 100;
		this.repetitions = 10000;
	}

	public SparseRowAllocPerfTest(int m, int n) {
		this.m = m;
		this.n = n;
		this.warmupRuns = 20;
		this.repetitions = 250;

	}

	public static void main(String[] args) {
		new SparseRowAllocPerfTest(2000, 10000).compareInitAndAlloc(1000);
		new SparseRowAllocPerfTest(10000, 2000).compareInitAndAlloc(100);
//		new SparseRowAllocPerfTest(100, 100000).compareInitAndAlloc(100);
//		new SparseRowAllocPerfTest(2000, 10000).compareInitAndAlloc(100);
//		new SparseRowAllocPerfTest(2000, 10000).compareInitAndAlloc(100);
//		new SparseRowAllocPerfTest(2000, 10000).compareInitAndAlloc(100);
//		new SparseRowAllocPerfTest(2000, 10000).compareInitAndAlloc(100);
//		new SparseRowAllocPerfTest(2000, 10000).compareInitAndAlloc(100);
//		new SparseRowAllocPerfTest(1000000, 100).compareInitAndAlloc(100);
	}

	public void testDenAndSpaAlloc() {
			String[] results;
			String[] sparseRes = new String[nVar.length];
			String[] denseRes = new String[nVar.length];
			for(int i = 0; i < nVar.length; i++) {
				getMatrices(0.1);
				results = compareAlloc(2000, n);
				sparseRes[i] = results[0];
				denseRes[i] = results[1];
			}
			logResults(sparseRes, true);
			logResults(denseRes, false);
	}

	public void compareInitAndAlloc(int numVec) {
		String[] results;
		String[] sparseRes = new String[sparsities.length];
		String[] denseRes = new String[sparsities.length];
		for(int i = 0; i < sparsities.length; i++) {
			getMatrices(sparsities[i]);
			results = testInitAndAllocDiff(numVec, n);
			sparseRes[i] = results[0];
			denseRes[i] = results[1];
			System.out.println("rows: " + m + " cols: "+ n + " sparsity: " + sparsities[i]);
			System.out.println("Allocation test: " + results[0]);
			System.out.println("Initialization test: " + results[1]);
		}
		logResults(sparseRes, true);
		logResults(denseRes, false);
	}

	public void runComparison() {
		TimingUtils.time(() -> new SparseRowVector(10000), 100);
		double[] initTime = TimingUtils.time(() -> new SparseRowVector(10000), 1000000);
		LibSpoofPrimitives.setupThreadLocalMemory(20, 10000);
		TimingUtils.time(() -> LibSpoofPrimitives.allocSparseVector(10000), 100);
		double[] ringTime = TimingUtils.time(() -> LibSpoofPrimitives.allocSparseVector(10000),1000000);
		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();
		System.out.println("Allocation test: " + TimingUtils.stats(ringTime));
		System.out.println("Initialization test: " + TimingUtils.stats(initTime));
	}

	public String[] compareAlloc(int numVectors, int len) {
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

	public String[] testInitAndAllocDiff(int numVectors, int len) {

		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len, -1);

		TimingUtils.time(() -> allocateVector(), warmupRuns);
		double[] allocResults = TimingUtils.time(() -> allocateVector(), repetitions);

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();

		TimingUtils.time(() -> initializeVector(), warmupRuns);
		double[] initResults = TimingUtils.time(() -> initializeVector(), repetitions);

		String sparseTime = TimingUtils.stats(allocResults).split("\\+-")[0];
		String denseTime = TimingUtils.stats(initResults).split("\\+-")[0];

		return new String[] {sparseTime, denseTime};
	}

	public void allocateVector() {
		for(int i = 0; i < m; i++) {
			if(!sparseInA.isEmpty(i))
				LibSpoofPrimitives.vectMultWrite(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
		}
	}

	public void initializeVector() {
		for(int i = 0; i < m; i++) {
			if(!sparseInA.isEmpty(i))
				LibSpoofPrimitives.vectMultWriteInit(n, sparseInA.values(i), scalar, sparseInA.indexes(i), sparseInA.pos(i), sparseInA.size(i));
		}
	}

	public void getMatrices(double sparsity) {
		double[][] A = TestUtils.generateTestMatrix(m, n, -5, 5, sparsity, 1251);
		double[][] B = TestUtils.generateTestMatrix(m, n, -5, 5, sparsity, 1345);
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

	public void logResults(String[] result, boolean sparse) {
		String currDate = LocalDateTime.now().format(DateTimeFormatter.ofPattern("ddMMyyyy_HHmmss"));
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(new FileWriter("C:\\Users\\tomok\\OneDrive - Technische Universität Berlin\\Bachelorarbeit\\performance\\results\\alloc_testing\\"
				+ "allocTest" + currDate + "_" + (sparse ? "alloc" : "init") + "_" + ".csv"));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		writer.printf("Repetitions: %1$2s, rl: %2$2s, cl: %3$2s%n", repetitions, m, n);
		writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", "Sparsity", "rows", "cols","time in ms");
		for(int i = 0; i < sparsities.length; i++) {
			writer.printf("%1$2s;%2$2s;%3$2s;%4$2s%n", sparsities[i], m, n, result[i]);
		}

		writer.flush();
		writer.close();
	}
}
