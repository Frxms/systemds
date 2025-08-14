package org.apache.sysds.performance.primitives;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.api.jmlc.PreparedScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.codegen.RowAggTmplTest;
import org.junit.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;

public class TypicalExpressionTest extends AutomatedTestBase {

	private static final Log LOG = LogFactory.getLog(RowAggTmplTest.class.getName());

	private static final String TEST_NAME = "expression";
	private static final String TEST_NAME1 = TEST_NAME+"1";
	private static final String TEST_NAME2 = TEST_NAME+"2";

	private static final String TEST_DIR = "performance/primitives/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TypicalExpressionTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;
	private final static double eps = 1e-8;
	private int rows = 500;
	private int cols = 1000;
	double[] sparsities = new double[] {1, 0.3333, 0.1111, 0.0333, 0.0111, 0.0033, 0.0011};
	int warmupRuns = 1;
	int repetitions = 2;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=2; i++)
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i, new String[] { String.valueOf(i) }) );
	}

	@Test
	public void testSparseNewImpl1CP() {runSparseExpression(TEST_NAME1, true, true, ExecType.CP);}

	@Test
	public void testOldSparseImpl1CP() {runSparseExpression(TEST_NAME1, true, false, ExecType.CP);}

	@Test
	public void testSparseNewImpl2CP() {runSparseExpression(TEST_NAME2, true, true, ExecType.CP);}

	@Test
	public void testOldSparseImpl2CP() {runSparseExpression(TEST_NAME2, true, false, ExecType.CP);}

	public static void main(String[] args) {
		TypicalExpressionTest t = new TypicalExpressionTest();
		t.setUpBase();
		t.runBenchmark(TEST_NAME2);
		t.tearDown();
	}

	public void runBenchmark(String testname) {
		setUp();
		logResults(runPerfTest(testname,  true, ExecType.CP), true);
		logResults(runPerfTest(testname, false, ExecType.CP), false);
	}

	public void runBenchmark(String testname, int rows, int cols) {
		setUp();
		this.rows = rows;
		this.cols = cols;
		logResults(runPerfTest(testname,  true, ExecType.CP), true);
		logResults(runPerfTest(testname, false, ExecType.CP), false);
	}

	private String[] runPerfTest(String testname, boolean sparseRowVec, ExecType et) {

		ExecMode platformOld = setExecMode(et);

		String[] resultTime = new String[sparsities.length];

		Connection conn = new Connection();

		try {

			getAndLoadTestConfiguration(testname);
			String HOME = SCRIPT_DIR + TEST_DIR;
			String script = DMLScript.readDMLScript(true, HOME + testname + ".dml");

			String[] inputs = new String[] {"A", "v", "B"};
			PreparedScript pscript = conn.prepareScript(script, inputs, new String[] {"S"});

			boolean oldSparse = DMLScript.SPARSE_INTERMEDIATE;
			DMLScript.SPARSE_INTERMEDIATE = sparseRowVec;

			for(int i = 0; i < sparsities.length; i++) {
				//generate random input matrices
				double[][] A = getRandomMatrix(rows, cols, 1, 31, sparsities[i], 1234);
				double[][] B = getRandomMatrix(rows, cols, 1, 31, sparsities[i], 5678);
				double[][] V = getRandomMatrix(rows, 1, 1, 31, sparsities[i], 9876);

				pscript.setMatrix("A", A);
				pscript.setMatrix("v", V);
				pscript.setMatrix("B", B);

				TimingUtils.time(() -> pscript.executeScript(), warmupRuns);
				double[] result = TimingUtils.time(() -> pscript.executeScript(), repetitions);

				resultTime[i] = TimingUtils.stats(result).split("\\+-")[0];

				System.out.println(TimingUtils.stats(result).split("\\+-")[0]);

				pscript.clearPinnedData();
			}

			DMLScript.SPARSE_INTERMEDIATE = oldSparse;

		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
		return resultTime;
	}

	private void runSparseExpression(String testname, boolean sparse, boolean sparseRowVec, ExecType et) {

		ExecMode platformOld = setExecMode(et);

		try {

			getAndLoadTestConfiguration(testname);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			if(sparseRowVec)
//				programArgs = new String[]{"-explain", "codegen", "-sparseIntermediate", "-args",
//					input("A"), input("B"), input("V"), output("S")};
				programArgs = new String[]{"-sparseIntermediate", "-args",
					input("A"), input("B"), input("V"), output("S")};
			else
//				programArgs = new String[]{"-explain", "codegen", "-args",
//					input("A"), input("B"), input("V"), output("S")};
				programArgs = new String[]{"-args",
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

	public void logResults(String[] result, boolean sparseInterm) {
		String currDate = LocalDateTime.now().format(DateTimeFormatter.ofPattern("ddMMyyyy_HHmmss"));
		PrintWriter writer = null;
		try {
			writer = new PrintWriter(new FileWriter("C:\\Users\\tomok\\OneDrive - Technische Universität Berlin\\Bachelorarbeit\\performance\\expression_testing\\"
				+ "expressionTest_"+ currDate + "_" + (sparseInterm ? "sparseInterm" : "denseInterm") + "_" + ".csv"));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		writer.printf("Repetitions: %1$2s, rl: %2$2s, cl: %3$2s%n", repetitions, rows, cols);
		writer.printf("%1$2s;%2$2s%n", "Sparsity", "time in ms");
		for(int i = 0; i < sparsities.length; i++) {
			writer.printf("%1$2s;%2$2s%n",  sparsities[i], result[i]);
		}

		writer.flush();
		writer.close();
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
