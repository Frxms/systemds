package org.apache.sysds.performance.primitives;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.ScriptExecutorUtils;
import org.apache.sysds.api.mlcontext.Script;
import org.apache.sysds.api.mlcontext.ScriptExecutor;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Explain;

import static org.apache.sysds.api.mlcontext.ScriptFactory.dml;

public class dmlExpressionsTest {
	private static final String DML_SCRIPT = "src/test/scripts/performance/primitives/expression1.dml";
	private static final String DML_CONF = "src/test/scripts/performance/primitives/SystemDS-config-codegen.xml";

	private static class Exec extends ScriptExecutor {
		Exec(DMLConfig conf) { super(conf); }
		public void initExecutionContext() { createAndInitializeExecutionContext(); }
	}

	public static void main(String[] args) throws Exception {
		int rows = 2000;
		int cols = 10000;
		double sparsity = 0.1;

		double[][] A = TestUtils.generateTestMatrix(rows, cols, 1, 31, sparsity, 1234);
		double[][] B = TestUtils.generateTestMatrix(rows, cols, 1, 31, sparsity, 5678);
		double[][] v = TestUtils.generateTestMatrix(rows, 1, 1, 31, sparsity, 9876);

		MatrixBlock AM = DataConverter.convertToMatrixBlock(A);
		MatrixBlock BM = DataConverter.convertToMatrixBlock(B);
		MatrixBlock vM = DataConverter.convertToMatrixBlock(v);

		String scriptStr = DMLScript.readDMLScript(true, DML_SCRIPT);

		Script script = dml(scriptStr).in("A", AM).in("B", BM).in("v", vM).out("S");

		Exec se = new Exec(new DMLConfig(DML_CONF));
		se.setStatistics(true);
		se.setStatisticsMaxHeavyHitters(10);
		se.setExplain(true);
		DMLScript.EXPLAIN = Explain.ExplainType.CODEGEN;
		se.compile(script);
		se.initExecutionContext();

		int warmup = 5;
		int runs = 20;
		TimingUtils.time(() -> ScriptExecutorUtils.executeRuntimeProgram(se, 10), warmup);
		double[] times = TimingUtils.time(() -> ScriptExecutorUtils.executeRuntimeProgram(se, 10), runs);
		System.out.println("Runtime: " + TimingUtils.stats(times));
	}
}
