package org.apache.sysds.performance.primitives;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.api.jmlc.PreparedScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Explain;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class ExpressionTest {

	private static final String TEST_NAME = "expression";
	private static final String TEST_NAME1 = TEST_NAME+"1";
	private static final String TEST_NAME2 = TEST_NAME+"2";
	private static final String TEST_NAME3 = TEST_NAME+"3";
	private static final String TEST_NAME4 = TEST_NAME+"4";
	private static final String TEST_NAME5 = TEST_NAME+"5";

	private static final String TEST_DIR = "./src/test/scripts/performance/primitives/";

	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;
	private final static double eps = 1e-8;
	private int rows = 1;
	private int cols = 100;
//	double[] sparsities = new double[] {1, 0.3333333, 0.1111111, 0.0333333, 0.0111111, 0.0033333, 0.0011111, 0.0003333, 0.0001111, 0.0000333, 0.0000111, 0.0000033, 0.0000011, 0.0000003, 0.0000001};
//	double[] sparsities = new double[] {1, 0.3333333, 0.1111111, 0.0333333, 0.0111111, 0.0033333, 0.0011111};
	double[] sparsities = new double[] {0.0111111};
	int warmupRuns = 20;
	int repetitions = 200;

	public ExpressionTest() {
	}

	public ExpressionTest(int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
	}

	public static void main(String[] args) {
//		new ExpressionTest(2000, 100000).runSparseBenchmark(TEST_NAME1);
//		new ExpressionTest(100000, 2000).runSparseBenchmark(TEST_NAME1);
//		new ExpressionTest(2000, 100000).runSparseBenchmark(TEST_NAME2);
//		new ExpressionTest(2000, 100000).runSparseBenchmark(TEST_NAME3);
//		new ExpressionTest(2000, 100000).runSparseBenchmark(TEST_NAME4);
//		new ExpressionTest(2000, 100000).runSparseBenchmark(TEST_NAME5);

		new ExpressionTest(2000, 100000).runDenseBenchmark(TEST_NAME1);
//		new ExpressionTest(100000, 2000).runDenseBenchmark(TEST_NAME1);
//		new ExpressionTest(2000, 100000).runDenseBenchmark(TEST_NAME2);
//		new ExpressionTest(2000, 100000).runDenseBenchmark(TEST_NAME3);
//		new ExpressionTest(2000, 100000).runDenseBenchmark(TEST_NAME4);
//		new ExpressionTest(10, 100 ).runDenseBenchmark(TEST_NAME5);
	}

	public void runSparseBenchmark(String testname) {
		logResults(runPerfTest(testname,  true, Types.ExecType.CP), true);
	}

	public void runDenseBenchmark(String testname) {
		logResults(runPerfTest(testname, false, Types.ExecType.CP), false);
	}

	public void runBenchmark(String testname, int rows, int cols) {
		this.rows = rows;
		this.cols = cols;
		logResults(runPerfTest(testname,  true, Types.ExecType.CP), true);
		logResults(runPerfTest(testname, false, Types.ExecType.CP), false);
	}

	private String[] runPerfTest(String testname, boolean sparseRowVec, Types.ExecType et) {
		System.out.println("Dimensions: " + rows + "x" + cols);
		String[] resultTime = new String[sparsities.length];

		DMLConfig cfg = new DMLConfig();
		cfg.setTextValue(DMLConfig.CODEGEN, "true");
		cfg.setTextValue(DMLConfig.CODEGEN_OPTIMIZER, "fuse_all");
		Connection conn = new Connection(cfg, CompilerConfig.ConfigType.CODEGEN_ENABLED);
//		Connection conn = new Connection(new DMLConfig(), CompilerConfig.ConfigType.CODEGEN_ENABLED);
		DMLScript.EXPLAIN = Explain.ExplainType.CODEGEN;
		boolean oldSparse = DMLScript.SPARSE_INTERMEDIATE;
		DMLScript.SPARSE_INTERMEDIATE = sparseRowVec;

		try {

			String HOME = TEST_DIR;
			String script = DMLScript.readDMLScript(true, HOME + testname + ".dml");

			String[] inputs = new String[] {"A", "v", "B"};
			PreparedScript pscript = conn.prepareScript(script, inputs, new String[] {"S"});

			for(int i = 0; i < sparsities.length; i++) {
				double[][] A = TestUtils.generateTestMatrix(rows, cols, 1, 31, sparsities[i], 1234);
				double[][] B = TestUtils.generateTestMatrix(rows, cols, 1, 31, sparsities[i], 5678);
				double[][] V = TestUtils.generateTestMatrix(rows, 1, 1, 31, sparsities[i], 9876);

				MatrixBlock AM = DataConverter.convertToMatrixBlock(A);
				MatrixBlock BM = DataConverter.convertToMatrixBlock(B);
				MatrixBlock vM = DataConverter.convertToMatrixBlock(V);

//				AM.denseToSparse();
//				BM.denseToSparse();
//				vM.denseToSparse();

				pscript.setMatrix("A", AM, true);
				pscript.setMatrix("B", BM, true);
				pscript.setMatrix("v", vM, true);

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
			conn.close();
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
		return resultTime;
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

}
