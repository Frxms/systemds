package org.apache.sysds.performance.primitives;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.codegen.RowAggTmplTest;
import org.junit.Test;

import java.io.File;
import java.util.HashMap;

import static org.apache.sysds.common.Types.ExecMode.SPARK;

public class TypicalExpressionTest extends AutomatedTestBase {

	private static final Log LOG = LogFactory.getLog(RowAggTmplTest.class.getName());

	private static final String TEST_NAME = "expression";
	private static final String TEST_NAME1 = TEST_NAME+"1";

	private static final String TEST_DIR = "performance/primitives/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TypicalExpressionTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

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

		ExecMode platformOld = setExecMode(et);

		try {

			getAndLoadTestConfiguration(testname);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			if(sparseRowVec)
				programArgs = new String[]{"-explain", "codegen", "-sparseIntermediate", "-args",
					input("A"), input("B"), input("V"), output("S")};
			else
				programArgs = new String[]{"-explain", "codegen", "-args",
					input("A"), input("B"), input("V"), output("S")};


			fullRScriptName = HOME + TEST_NAME1 + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			//get a random matrix of values with
			double[][] A = getRandomMatrix(rows, cols, 1, 31, sparse ? sparsity2 : sparsity1, 1234);
			double[][] B = getRandomMatrix(rows, cols, 1, 31, sparse ? sparsity2 : sparsity1, 5678);
			double[][] V = getRandomMatrix(rows, 1, 1, 31, sparse ? sparsity2 : sparsity1, 9876);
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);
			writeInputMatrixWithMTD("V", V, true);

			//run tests
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("S");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R", true);
		}
		finally {
			resetExecMode(platformOld);
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}

	}

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		LOG.debug("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
