package org.apache.sysds.performance.primitives;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

import static org.apache.sysds.common.Types.ExecMode.SPARK;

public class TypicalExpressionTest extends AutomatedTestBase {

	private static final String TEST_NAME = "expression";
	private static final String TEST_NAME1 = TEST_NAME+"1";

	private static final String TEST_DIR = "performance/primitives/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TypicalExpressionTest.class.getSimpleName() + "/";

	private final static int rows = 500;
	private final static int cols = 1000;
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;
	private final static double eps = 1e-10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=1; i++)
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i, new String[] { String.valueOf(i) }) );
	}

	@Test
	public void testSparseNewImplCP() {runSparseExpression(TEST_NAME1, true, true, ExecType.CP);}

	@Test
	public void testOldSparseImplCP() {runSparseExpression(TEST_NAME1, true, false, ExecType.CP);}

	private void runSparseExpression(String testname, boolean sparse, boolean sparseRowVec, ExecType et) {

		ExecMode platformOld = rtplatform;

		switch( et ){
			case SPARK: rtplatform = SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {

			setOutputBuffering(true);
			String TEST_NAME = testname;
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			if(sparseRowVec)
				programArgs = new String[]{"-explain", "codegen", "-sparseIntermediate", "-args",
					input("A"), input("B"), input("v"), output("C")};
			else
				programArgs = new String[]{"-explain", "codegen", "-args",
					input("A"), input("B"), input("v"), output("C")};


			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			//get a random matrix of values with
			double[][] A = getRandomMatrix(rows, cols, 1, 31, sparse ? sparsity1 : sparsity2, 1234);
			double[][] B = getRandomMatrix(rows, cols, 1, 31, sparse ? sparsity1 : sparsity2, 5678);
			double[][] V = getRandomMatrix(1, cols, 1, 31, sparse ? sparsity1 : sparsity2, 9876);
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);
			writeInputMatrixWithMTD("V", V, true);

			//run tests
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R", true);
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			rtplatform = platformOld;
		}

	}

}
