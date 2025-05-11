package org.apache.sysds.test.component.codegen;

import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

/**
 * This is the component test for the ring buffer used in LibSpoofPrimitives.java,
 * that allocates a vector with a certain size.
 * Every allocation method is tested to achieve the needed coverage.
 */
public class SparseVectorAllocTest extends AutomatedTestBase
{
	double[] val1 = new double[]{1.5, 5.7, 9.1, 3.7, 5.3};
	double[] val2 = new double[]{9.6, 7.1, 2.7};
	int[] indexes1 = new int[]{3, 7, 14, 20, 81};
	int[] indexes2 = new int[]{20, 30, 90};

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testBasicAllocationSameLen() {
		testBasicSparseVectorAllocation(1, 10, 10);
	}

	@Test
	public void testBasicAllocationLongerExp() {
		testBasicSparseVectorAllocation(1, 10, 15);
	}

	@Test
	public void testBasicAllocationShorterExp() {
		testBasicSparseVectorAllocation(1, 10, 7);
	}

	@Test
	public void testAllocationWithValues1() {
		testSparseVectorWithValues(1, 10, val1.length,val1, indexes1);
	}

	@Test
	public void testAllocationWithValues2() {
		testSparseVectorWithValues(1, 10, val2.length, val2, indexes2);
	}

	@Test
	public void testAllocationWithValues3() {
		testSparseVectorWithValues(1, 10, 3, val2, indexes2);
	}

	@Test
	public void testAllocationWithIndexes1() {
		testSparseVectorWithIndexes(1, 10, indexes1.length, indexes1);
	}

	@Test
	public void testAllocationWithIndexes2() {
		testSparseVectorWithIndexes(1, 10, indexes2.length, indexes2);
	}

	@Test
	public void testAllocationWithIndexes3() {
		testSparseVectorWithIndexes(1, 10, 3, indexes1);
	}

	@Test
	public void testVectorReuse1() {
		testBufferReuse(1, 10, -1, 5, val1, indexes1);
	}

	@Test
	public void testVectorReuse2() {
		testBufferReuse(1, 10, -1, 5, val2, indexes2);
	}

	public void testBasicSparseVectorAllocation(int numVectors, int len, int expLen) {
		//test the basic allocation of an empty vector
		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len, -1);
		SparseRowVector sparseVec = LibSpoofPrimitives.allocSparseVector(expLen);

		Assert.assertEquals("Vector capacity should be initialized correctly", expLen, sparseVec.capacity());
		Assert.assertEquals("Vector size should be initialized with 0", 0, sparseVec.size());

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();
	}

	public void testSparseVectorWithValues(int numVectors, int len, int expLen,double[] values, int[] indexes) {
		//test the allocation of a vector with preset values
		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len, -1);

		SparseRowVector sparseVec = LibSpoofPrimitives.allocSparseVector(values.length, values, indexes);

		Assert.assertEquals("Vector size should match input array length", values.length, sparseVec.size());
		Assert.assertEquals("Value array should match input array", values, sparseVec.values());

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();
	}

	public void testSparseVectorWithIndexes(int numVectors, int len, int expLen,int[] indexes) {
		//test allocation of a vector with preset indexes
		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len, -1);

		SparseRowVector sparseVec = LibSpoofPrimitives.allocSparseVector(indexes.length, indexes);

		Assert.assertEquals("Vector size should match input array length", indexes.length, sparseVec.size());
		Assert.assertArrayEquals("Indexes array should match input array", indexes, sparseVec.indexes());

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();
	}

	public void testBufferReuse(int numVectors, int len1, int len2, int expLen, double[] values, int[] indexes) {
		//test the reuse of the vectors in the ring buffer
		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len1, len2);

		//allocate first vector
		SparseRowVector vec1 = LibSpoofPrimitives.allocSparseVector(expLen);
		vec1.set(0, 1.0);
		vec1.set(2, 2.0);

		//allocate second vector
		SparseRowVector vec2 = LibSpoofPrimitives.allocSparseVector(expLen);

		Assert.assertEquals("Reused vector should be reset to size 0", 0, vec2.size());

		//allocate third vector with values
		SparseRowVector vec3 = LibSpoofPrimitives.allocSparseVector(values.length, values, indexes);

		Assert.assertEquals("Vector size should match input", values.length, vec3.size());
		Assert.assertEquals("Value array should match input array", values, vec3.values());

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();
	}

}
