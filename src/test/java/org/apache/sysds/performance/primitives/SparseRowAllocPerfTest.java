package org.apache.sysds.performance.primitives;

import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.data.SparseRowVector;

public class SparseRowAllocPerfTest {

	public static String[] testAlloc(int numVectors, int len, int repetitions, int warmupRuns) {

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
		double bval = 5;
		int[] aix = {1,2,3};
		double[] a = {1.2, 3.4, 7};
		int ai = 0;
		int alen = 3;
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = a[j]*bval;
		}
		c.setSize(alen);
		System.out.println(c);
	}

	public static void initializeVector(int len) {
		SparseRowVector c = new SparseRowVector(len);
		double bval = 5;
		int[] aix = {1,2,3};
		double[] a = {1.2, 3.4, 7};
		int ai = 0;
		int alen = 3;
		int[] indexes = c.indexes();
		double[] values = c.values();
		for(int j = 0; j < ai+alen; j++) {
			indexes[j] = aix[j];
			values[j] = a[j]*bval;
		}
		c.setSize(alen);
		System.out.println(c);
	}

	public static void main(String[] args) {
		SparseRowAllocPerfTest.testAlloc(3, 1000, 10000000, 100);
	}
}
